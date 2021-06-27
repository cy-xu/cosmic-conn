const MAIN_IMAGE_ID = 'main-img'
const ZOOM_WINDOW_ID = 'zoom-window'
const ZOOMED_FRAME_IMAGE_ID = 'frame-zoom-img'
const ZOOMED_MASK_IMAGE_ID = 'mask-zoom-img'
const PIXEL_VALUE_INDICATOR_CLASS = 'pixel-value-indicator'

const ZOOMED_WINDOW_MINIMUM_PIXEL = 5
const ZOOMED_WINDOW_MAXIMUM_PIXEL = 500
const ZOOMED_WINDOW_PIXEL_STEP = 10

const ZOOMED_WINDOW_OUTLINE_WIDTH = 2
const ZOOMED_WINDOW_OUTLINE_RED = 0
const ZOOMED_WINDOW_OUTLINE_GREEN = 255 
const ZOOMED_WINDOW_OUTLINE_BLUE = 180

const ZOOMED_WINDOW_SLOTDOWN_THRESHOLD = 0.35


class ZoomWindow {
    constructor(image_width, image_height) {
        this.image_width = image_width
        this.image_height = image_height
        this.window_left_top_x = 0
        this.window_left_top_y = 0
        this.init_from_image_dimension(image_width, image_height)
    }

    init_from_image_dimension(image_width, image_height) {
        this.image_width = image_width
        this.image_height = image_height
        this.smallest_window_size = ZOOMED_WINDOW_MINIMUM_PIXEL
        this.largest_window_size = Math.min(image_width, image_height) * 0.40
        this.current_window_size = this.largest_window_size / 2
        this.window_size_change_step = (this.largest_window_size - this.smallest_window_size) / 50
    }

    get_window_position_and_size() {
        // window left top coordinate and the window size are represented by pxiels on the image
        return [this.window_left_top_x, this.window_left_top_y, this.current_window_size]
    }

    get_window_center() {
        let center_x = this.window_left_top_x + this.current_window_size / 2
        let center_y = this.window_left_top_y + this.current_window_size / 2
        return [center_x, center_y]
    }
    
    increase_window_size() {
        let [prev_center_x, prev_center_y] = this.get_window_center()
        let size = this.current_window_size
        let range = this.largest_window_size - this.smallest_window_size
        let bottom = size - this.smallest_window_size;
        size += this.window_size_change_step * ((bottom < range * ZOOMED_WINDOW_SLOTDOWN_THRESHOLD) ? (bottom / (range * ZOOMED_WINDOW_SLOTDOWN_THRESHOLD)) : 1);
        this.current_window_size = this.#regulate_window_size(size)
        this.#update_window_position(prev_center_x , prev_center_y)
    }

    decrease_window_size() {
        let [prev_center_x, prev_center_y] = this.get_window_center()
        let size = this.current_window_size
        let range = this.largest_window_size - this.smallest_window_size
        let bottom = size - this.smallest_window_size;
        size -= this.window_size_change_step * ((bottom < range * ZOOMED_WINDOW_SLOTDOWN_THRESHOLD) ? (bottom / (range * ZOOMED_WINDOW_SLOTDOWN_THRESHOLD)) : 1);
        this.current_window_size = this.#regulate_window_size(size)
        this.#update_window_position(prev_center_x , prev_center_y)
    }

    #regulate_window_size(size) {
        if (size < this.smallest_window_size)
            size = this.smallest_window_size
        else if (size > this.largest_window_size)
            size = this.largest_window_size
        return size
    }

    update_window_position_relative_to_self_center(x_percent, y_percent) {
        let x_offset = x_percent * this.current_window_size
        let y_offset = y_percent * this.current_window_size
        let center_x = this.window_left_top_x + this.current_window_size / 2 + x_offset
        let center_y = this.window_left_top_y + this.current_window_size / 2 + y_offset
        this.#update_window_position(center_x, center_y)
    }

    update_window_position_relative_to_image(x_percent, y_percent) {
        let center_x = x_percent * this.image_width
        let center_y = y_percent * this.image_height
        this.#update_window_position(center_x, center_y)   
    }

    #update_window_position(center_x, center_y) {
        const half_window_size = this.current_window_size / 2
        const left = center_x - half_window_size
        const right = center_x + half_window_size
        const top = center_y - half_window_size
        const bottom = center_y + half_window_size
        let [left_top_x, left_top_y] = [left, top]
        
        // clamp window range
        if (left < 0) {
            left_top_x = 0
        }
        else if (right > this.image_width) { 
            left_top_x = this.image_width - this.current_window_size
        }

        if (top < 0) {
            left_top_y = 0
        }
        else if (bottom > this.image_height) {
            left_top_y = this.image_height - this.current_window_size
        }

        this.window_left_top_x = Math.max(Math.floor(left_top_x), 0)
        this.window_left_top_y = Math.max(Math.floor(left_top_y), 0)
    }
}

class ImagePanelView {
    constructor(context) {
        this.context = context
        this.zoom_window = new ZoomWindow(0, 0)
        this.frame_canvas = null
        this.mask_canvas = null

        this.current_main_image_type = 'frame'

        // the main image display areas
        this.main_image = document.getElementById(MAIN_IMAGE_ID)
        this.zoomed_frame_image = document.getElementById(ZOOMED_FRAME_IMAGE_ID)
        this.zoomed_mask_image = document.getElementById(ZOOMED_MASK_IMAGE_ID)
        this.zoom_window_frame = document.getElementById(ZOOM_WINDOW_ID)

        this.#hookup_callbacks()
    }

    #hookup_callbacks() {
        $(this.main_image).click((event) => {
            this.#main_image_mouseclick_callback(event)
        })

        $(this.zoomed_frame_image).click((event) => {
            this.#zoom_image_mouseclick_callback(event)
        })

        $(this.zoomed_mask_image).click((event) => {
            this.#zoom_image_mouseclick_callback(event)
        })

        $(this.zoomed_frame_image).mousemove((event) => {
            this.#zoom_image_mousemove_callback(event)
        })

        $(this.zoomed_mask_image).mousemove((event) => {
            this.#zoom_image_mousemove_callback(event)
        })

        $(this.main_image).on('wheel', (event) => {
            this.#main_image_mousescroll_callback(event)
        })

        $(this.zoomed_frame_image).on('wheel', (event) => {
            this.#zoom_image_mousescroll_callback(event)
        })

        $(this.zoomed_mask_image).on('wheel',(event) => {
            this.#zoom_image_mousescroll_callback(event)
        })
    }

    #zoom_image_mousemove_callback(event) {
        let offset = $(event.target).offset()
        let x = event.pageX - offset.left
        let y = event.pageY - offset.top
        let total_width = event.target.width
        let total_height = event.target.height
        let x_percent = x/total_width
        let y_percent = y/total_height
        let [left_top_x_on_image, left_top_y_on_image, window_width] = this.zoom_window.get_window_position_and_size()
        
        let sample_x = (left_top_x_on_image + x_percent * window_width) * 1.0
        let sample_y = (left_top_y_on_image + y_percent * window_width) * 1.0
        
        if (sample_x === undefined || sample_y === undefined)
            return
        
        try {
            let value = this.context.raw_image_model.value_at(sample_x, sample_y)
            let output = `pixel: (${sample_x.toFixed(2)}, ${sample_y.toFixed(2)}) ` + `value: ${value.toFixed(5)}` 
            $('.' + PIXEL_VALUE_INDICATOR_CLASS).text(output)
        }
        catch {}
    }

    #zoom_image_mouseclick_callback(event) {
        let offset = $(event.target).offset()
        let x = event.pageX - offset.left
        let y = event.pageY - offset.top
        let total_width = event.target.width
        let total_height = event.target.height
        let x_percent = x/total_width
        let y_percent = y/total_height
        let center_offset_percent_x = x_percent - 0.5
        let center_offset_percent_y = y_percent - 0.5
        if (Math.abs(center_offset_percent_x) < 0.005 && Math.abs(center_offset_percent_y) < 0.005)
            return
        this.zoom_window.update_window_position_relative_to_self_center(center_offset_percent_x, center_offset_percent_y)
        this.refresh_zoomed_images()
        this.refresh_zoom_window()
    }

    #zoom_image_mousescroll_callback(event) {
        if (event.originalEvent.deltaY < 0)
            this.zoom_window.decrease_window_size()
        else if (event.originalEvent.deltaY > 0)
            this.zoom_window.increase_window_size()
        this.refresh_zoomed_images()
        this.refresh_zoom_window()
    }

    #main_image_mouseclick_callback(event) {
        this.#update_zoom_window_based_on_mouse_location_event_on_main(event)
    }
    
    #main_image_mousescroll_callback(event) {
        if (event.originalEvent.deltaY < 0)
            this.zoom_window.increase_window_size()
        else if (event.originalEvent.deltaY > 0)
            this.zoom_window.decrease_window_size()
        this.#update_zoom_window_based_on_mouse_location_event_on_main(event)
    }

    #update_zoom_window_based_on_mouse_location_event_on_main(event) {
        let offset = $(event.target).offset()
        let x = event.pageX - offset.left
        let y = event.pageY - offset.top
        let total_width = event.target.width
        let total_height = event.target.height
        this.update_zoom_window_position(x/total_width, y/total_height)
    }

    update_zoom_window_position(x_percent, y_percent) {
        this.zoom_window.update_window_position_relative_to_image(x_percent, y_percent)
        this.refresh_zoomed_images()
        this.refresh_zoom_window()
    }

    set_current_main_image_type(type) {
        this.current_main_image_type = type
    }

    set_new_zoom_window_based_on_image_size(width, height) {
        this.zoom_window = new ZoomWindow(width, height)
    }

    set_frame_image_pixels_from_grayscale_array(width, height, grayscale_array) {
        this.frame_canvas = this.#construct_canvas_from_grayscale_array(width, height, grayscale_array)
    }

    set_mask_image_pixels_from_grayscale_array(width, height, grayscale_array) {
        this.mask_canvas = this.#construct_canvas_from_grayscale_array(width, height, grayscale_array)
    }

    #construct_canvas_from_grayscale_array(width, height, grayscale_array) {
        let canvas = document.createElement('canvas')
        canvas.setAttribute('width', width)
        canvas.setAttribute('height', height)
        let context = canvas.getContext('2d')
        let image_data = context.getImageData(0, 0, canvas.width, canvas.height)

        const pixel_number = width * height
        for (let i = 0; i < pixel_number; i++) {
            let v = grayscale_array[i]
            image_data.data[4*i] = v
            image_data.data[4*i + 1] = v
            image_data.data[4*i + 2] = v
            image_data.data[4*i + 3] = 255
        }

        // let src = cv.matFromArray(height, width, cv.CV_8U, grayscale_array)
        // let dst = new cv.Mat()
        // cv.cvtColor(src, dst, cv.COLOR_GRAY2BGRA, 0)
        // let image_data = new ImageData(new Uint8ClampedArray(dst.data), width, height)

        context.putImageData(image_data, 0, 0)
        return canvas
    }

    /**
     * Redraw the zoom window reactangle on the main image thumbnail
     * @returns 
     */
    refresh_zoom_window() {
        let [x, y, width] = this.zoom_window.get_window_position_and_size()
        if (this.frame_canvas == null || width == 0) {
            $(this.zoom_window_frame).css('border-style', '')
            return
        }

        let dom_total_width = this.main_image.width
        let dom_total_height = this.main_image.height
        let [x_on_image, y_on_image, window_width_on_image] =
            this.zoom_window.get_window_position_and_size()
        let x_percent = x_on_image / this.frame_canvas.width
        let y_percent = y_on_image / this.frame_canvas.height
        
        let window_dom_left_top_x = dom_total_width * x_percent
        let window_dom_left_top_y = dom_total_height * y_percent
        let window_dom_size = window_width_on_image / this.frame_canvas.width * dom_total_width
        
        let r = ZOOMED_WINDOW_OUTLINE_RED
        let g = ZOOMED_WINDOW_OUTLINE_GREEN
        let b = ZOOMED_WINDOW_OUTLINE_BLUE
        $(this.zoom_window_frame).css('left', Math.floor(window_dom_left_top_x))
        $(this.zoom_window_frame).css('top', Math.floor(window_dom_left_top_y))
        $(this.zoom_window_frame).css('width', Math.floor(window_dom_size))
        $(this.zoom_window_frame).css('height', Math.floor(window_dom_size))
        $(this.zoom_window_frame).css('border-width', ZOOMED_WINDOW_OUTLINE_WIDTH + 'px')
        $(this.zoom_window_frame).css('border-style', 'solid')
        $(this.zoom_window_frame).css('border-color', `rgb(${r},${g},${b}`)
    }
    
    post_process_mask_image_dilation(dilation_value) {
        let src = cv.imread(this.mask_canvas)
        let dst = new cv.Mat();
        let M = cv.Mat.ones(3, 3, cv.CV_8U);
        let anchor = new cv.Point(-1, -1);
        // You can try more different parameters
        cv.dilate(src, dst, M, anchor, dilation_value, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
        cv.imshow(this.mask_canvas, dst);
        src.delete(); dst.delete(); M.delete();
    }


    refresh_main_image_with_frame() {
        if (this.frame_canvas == null)
            return
        this.main_image.src = this.frame_canvas.toDataURL()
    }

    /**
     * Redraw both the main frame image and the zoom image.
     * @returns 
     */
    refresh_zoomed_images() {
        let [x, y, width] = this.zoom_window.get_window_position_and_size()
        if (this.frame_canvas == null || width == 0)
            return
        let frame_canvas_context = this.frame_canvas.getContext('2d')
        let mask_canvas_context = this.mask_canvas.getContext('2d')
        let zoomed_frame_data = frame_canvas_context.getImageData(x, y, width, width)
        let zoomed_mask_data = mask_canvas_context.getImageData(x, y, width, width)
        let zoomed_frame_data_url = this.#convert_image_data_to_dataurl(width, width, zoomed_frame_data)
        let zoomed_mask_data_url = this.#convert_image_data_to_dataurl(width, width, zoomed_mask_data)
        this.zoomed_frame_image.src = zoomed_frame_data_url
        this.zoomed_mask_image.src = zoomed_mask_data_url
    }

    #convert_image_data_to_dataurl(image_width, image_height, image_data) {
        let temp_canvas = document.createElement('canvas')
        temp_canvas.setAttribute('width', image_width)
        temp_canvas.setAttribute('height', image_height)
        temp_canvas.getContext('2d').putImageData(image_data, 0, 0)
        return temp_canvas.toDataURL()
    }
}