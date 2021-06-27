const SCALE_SELECT_MENU_ID = 'scale-select-menu'
const SCALE_SELECT_MENU_ITEM_CLASS = 'dropdown-item'
const SCALE_SELECT_DROPDOWN_BUTTON_ID = 'scale-select-dropdown-button'
const SCALE_CURRENT_METHOD_LABEL = 'curr-scale-method-label'

const GRAYSCALE_CLAMP_MIN_INPUT = 'scale-min-input'
const GRAYSCALE_CLAMP_MAX_INPUT = 'scale-max-input'

const IMAGE_TOGGLER_ID = 'image-toggle-area'
const FRAME_IMAGE_TOGGLER_ID = 'frame-img-button'
const MASK_IMAGE_TOGGLER_ID = 'mask-img-button'

const DILATION_VALUE_INPUT_ID = 'dilation-value'
const DILATION_PLUS_BUTTON_ID = 'dilation-plus-button'
const DILATION_MINUS_BUTTON_ID = 'dilation-minus-button'

const THRESHOLD_VALUE_INPUT_ID = 'threshold-value'
const THRESHOLD_PLUS_BUTTON_ID = 'threshold-plus-button'
const THRESHOLD_MINUS_BUTTON_ID = 'threshold-minus-button'

const SCALE_SLIDER_ID = 'scale-slider'
const GRAYSCALE_COLOR_MAP_RANGE_CONTROL_ID = 'black-white-map-control'
const BLACK_VALUE_INPUT_ID = 'black-value-input'
const WHITE_VALUE_INPUT_ID = 'white-value-input'

class ControlPanel {
    constructor(controller) {
        this.context = controller

        this.scale_select_menu = document.getElementById(SCALE_SELECT_MENU_ID)
        this.scale_current_selected = document.getElementById(SCALE_CURRENT_METHOD_LABEL)
        this.clamp_min_input = document.getElementById(GRAYSCALE_CLAMP_MIN_INPUT)
        this.clamp_max_input = document.getElementById(GRAYSCALE_CLAMP_MAX_INPUT)

        this.dilation_value_input = document.getElementById(DILATION_VALUE_INPUT_ID)
        this.dilation_plus_button = document.getElementById(DILATION_PLUS_BUTTON_ID)
        this.dilation_minus_button = document.getElementById(DILATION_MINUS_BUTTON_ID)

        this.threshold_value_input = document.getElementById(THRESHOLD_VALUE_INPUT_ID)
        this.threshold_plus_button = document.getElementById(THRESHOLD_PLUS_BUTTON_ID)
        this.threshold_minus_button = document.getElementById(THRESHOLD_MINUS_BUTTON_ID)

        this.color_map_control = document.getElementById(GRAYSCALE_COLOR_MAP_RANGE_CONTROL_ID)
        this.black_value_input = document.getElementById(BLACK_VALUE_INPUT_ID)
        this.white_value_input = document.getElementById(WHITE_VALUE_INPUT_ID)

        this.#hookup_callbacks()

        this.reset_control_parameters()
    }

    reset_control_parameters() {
        this.scale_current_selected.innerHTML = 'zscale'
        this.clamp_min_input.value = 1
        this.clamp_max_input.value = 5000
        this.dilation_value_input.value = 0
        this.black_value_input.value = 0
        this.white_value_input.value = 255
        this.threshold_value_input.value = 0.5.toFixed(2)
    }

    set_scale_clamp_range(min, max) {
        this.clamp_min_input.value = min
        this.clamp_max_input.value = max
    }

    get_current_scale_method_name() {
        let current_scale_method = this.scale_current_selected.innerHTML
        return current_scale_method.toLowerCase()
    }

    get_scale_clamp_range() {
        let min = parseFloat(this.clamp_min_input.value)
        let max = parseFloat(this.clamp_max_input.value)
        return [min, max]
    }

    get_black_white_points_percent() {
        let black_percent = parseFloat(this.black_value_input.value) / 255
        let white_percent = parseFloat(this.white_value_input.value) / 255
        return [black_percent, white_percent]
    }

    get_dilation_value() {
        let dilation_value = parseInt(this.dilation_value_input.value)
        return dilation_value
    }

    get_threshold_value() {
        let threshold_value = parseFloat(this.threshold_value_input.value).toFixed(2)
        return threshold_value
    }

    #hookup_callbacks() {
        $(this.scale_select_menu).find('.' + SCALE_SELECT_MENU_ITEM_CLASS).click((event) => {
            this.#scale_method_selected_callback(event)
        })

        $(this.clamp_min_input).change((event) => {
            this.#clamp_range_input_changed_callback(event)
        })

        $(this.clamp_max_input).change((event) => {
            this.#clamp_range_input_changed_callback(event)
        })

        $(this.color_map_control).find('input').change((event) => {
            this.#color_map_value_changed_callback(event)
        })

        $(this.dilation_value_input).change((event) => {
            this.#dilation_value_changed_callback(event)
        })

        $(this.dilation_plus_button).click((event) => {
            this.#dilation_increase_clicked_callback(event)
        })

        $(this.dilation_minus_button).click((event) => {
            this.#dilation_decrease_clicked_callback(event)
        })

        $(this.threshold_value_input).change((event) => {
            this.#threshold_value_input_changed_callback(event)
        })

        $(this.threshold_plus_button).click((event) => {
            this.#threshold_plus_button_clicked_callback(event)
        })

        $(this.threshold_minus_button).click((event) => {
            this.#threshold_minus_button_clicked_callback(event)
        })
    }

    // callbacks
    // Scale Method Control
    #scale_method_selected_callback(event) {
        let selected_scale_method = event.target.innerHTML
        this.scale_current_selected.innerHTML = selected_scale_method

        try {
            this.context.refresh_frame_view(FramePipeStage.CURVE_STAGE)
            this.context.refresh_images_and_zoom_window()
        } catch { }
    }

    // Clamp Range Control
    #clamp_range_input_changed_callback(event) {
        try {
            this.context.refresh_frame_view(FramePipeStage.CLAMP_STAGE)
            this.context.refresh_images_and_zoom_window()
        } catch { }
    }

    // Color Map Control
    #color_map_value_changed_callback(event) {
        let black_value = parseInt(this.black_value_input.value)
        let white_value = parseInt(this.white_value_input.value)
        if (black_value < 0) {
            this.black_value_input.value = 0
            black_value = 0
        }
        if (white_value > 255) {
            this.white_value_input.value = 255
            white_value = 255
        }

        if (black_value > white_value) {
            this.white_value_input.value = black_value
        }

        try {
            this.context.refresh_frame_view(FramePipeStage.RENDER_STAGE)
            this.context.refresh_images_and_zoom_window()
        } catch { }
    }

    // Dilation Control
    #dilation_increase_clicked_callback(event) {
        let val = parseInt(this.dilation_value_input.value)
        this.dilation_value_input.value = val + 1
        this.#dilation_value_changed_callback(event)
    }

    #dilation_decrease_clicked_callback(event) {
        let val = parseInt(this.dilation_value_input.value)
        this.dilation_value_input.value = val - 1
        this.#dilation_value_changed_callback(event)
    }

    #dilation_value_changed_callback(event) {
        let dilation_value = this.dilation_value_input.value
        if (!dilation_value || parseInt(dilation_value) < 0) {
            this.dilation_value_input.value = '0'
            dilation_value = 0
        }

        try {
            this.context.refresh_mask_view(MaskPipeStage.DILATION_STAGE)
            this.context.refresh_zoom_images()
        } catch (e) {
            console.log('Refresh Failed. Details: ', e)
        }
    }

    // Threshold Control
    #threshold_plus_button_clicked_callback(event) {
        let val = parseFloat(this.threshold_value_input.value)
        this.threshold_value_input.value = val + 0.1
        this.#threshold_value_input_changed_callback(event)
    }

    #threshold_minus_button_clicked_callback(event) {
        let val = parseFloat(this.threshold_value_input.value)
        this.threshold_value_input.value = val - 0.1
        this.#threshold_value_input_changed_callback(event)
    }

    #threshold_value_input_changed_callback(event) {
        let value = parseFloat(this.threshold_value_input.value)
        this.threshold_value_input.value = ((value > 1) ? 1 : (value < 0) ? 0 : value).toFixed(2)

        try {
            this.context.refresh_mask_view(MaskPipeStage.BINARY_STAGE)
            this.context.refresh_zoom_images()
        } catch {
            console.log("Failed. Image not loaded.")
        }
    }
}