const THUMBNAIL_CONTAINER_ID = "thumbnail-container"
const THUMBNAIL_TEMPLATE_ID = "thumbnail-element-template"

class ThumbnailBar {
    constructor(controller) {
        this.context = controller
        this.thumbnail_container = document.getElementById(THUMBNAIL_CONTAINER_ID)
        this.thumbnail_lists = []

        // Load the splide library
        var thumbnails = new Splide('#thumbnail-bar', {
            fixedWidth: 100,
            gap: '1rem',
            pagination: false,
            trimSpace: true,
        });
        thumbnails.mount();
    }

    clear_thumbnails() {
        $(this.thumbnail_container).empty()
    }

    // takes ImageDataWrapper in the ImageModel.js
    append_thumbnails(mask_image_wrapper, coord_list, patch_size) {
        // initialize a temprary canvas for croping the data
        const [width, height] = [mask_image_wrapper.width, mask_image_wrapper.height]
        let image_array = mask_image_wrapper.image_array

        let canvas = document.createElement('canvas')
        canvas.setAttribute('width', width)
        canvas.setAttribute('height', height)
        let context = canvas.getContext('2d')
        let image_data = context.getImageData(0, 0, canvas.width, canvas.height)

        const pixel_number = width * height
        for (let i = 0; i < pixel_number; i++) {
            let v = image_array.get(i) > 0.5 ? 255 : 0
            image_data.data[4 * i] = v
            image_data.data[4 * i + 1] = v
            image_data.data[4 * i + 2] = v
            image_data.data[4 * i + 3] = 255
        }

        context.putImageData(image_data, 0, 0)

        // loop through the thumbnails to create image dom.
        for (let i = 0; i < coord_list.length; i++) {
            let [x, y] = coord_list[i];
            let img_data = context.getImageData(x, y, patch_size, patch_size)

            let temp_canvas = document.createElement('canvas')
            temp_canvas.setAttribute('width', patch_size)
            temp_canvas.setAttribute('height', patch_size)
            temp_canvas.getContext('2d').putImageData(img_data, 0, 0)
            let data_url = temp_canvas.toDataURL()

            let thumbnail = this.#create_thumbnail_dom(data_url)
            thumbnail = this.#attach_thumbnail_click_callback(thumbnail, x, y, patch_size)

            $(this.thumbnail_container).append($(thumbnail))
        }
    }

    #create_thumbnail_dom(data_url) {
        let template = $('#' + THUMBNAIL_TEMPLATE_ID).clone().html();
        let thumbnail = $(template)
        thumbnail.find("img").attr("src", data_url)
        return thumbnail
    }

    #attach_thumbnail_click_callback(thumbnail, x, y, patch) {
        thumbnail.find("img").click(() => {
            let xx = x + patch / 2
            let yy = y + patch / 2
            let [width, height] = this.context.raw_image_model.image_dimension
            this.context.image_view.update_zoom_window_position(xx / width, yy / height)
        });
        return thumbnail
    }

    refresh_thumbnails() {
        var thumbnails = new Splide('#thumbnail-bar', {
            fixedWidth: 100,
            gap: '1rem',
            pagination: false,
            trimSpace: true,
        });
        thumbnails.mount();
    }

}