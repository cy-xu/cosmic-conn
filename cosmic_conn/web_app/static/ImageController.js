// Image controller contains all the classes related to the two main preview windows

const FramePipeStage = {
    RAW_STAGE: 0,
    CLAMP_STAGE: 1,
    SIGMA_STAGE: 2,
    CURVE_STAGE: 3,
    RENDER_STAGE: 4
}

const MaskPipeStage = {
    RAW_STAGE: 0,
    BINARY_STAGE: 1,
    DILATION_STAGE: 2,
    RENDER_STAGE: 3,
    PIXEL_STAGE: 4
}

class PixelRecord {
    constructor() {
        this.record = {}
    }

    #tokey(pixel) {
        return `${pixel.x},${pixel.y}`
    }
    
    exist(pixel) {
        let key = this.#tokey(pixel)
        return key in this.record
    }

    add(pixel) {
        if (this.exist(pixel))
            return
        let key = this.#tokey(pixel)
        this.record[key] = pixel
    }

    remove(pixel) {
        if (!this.exist(pixel))
            return
        let key = this.#tokey(pixel)
        delete this.record[key]
    }

    tolist() {
        return Object.values(this.record)
    }

    clear() {
        for (const key in this.record) {
            delete this.record[key]
        }
    }
}

class ImageController {
    constructor() {
        // Dilation Edited Pixels
        // this.dilation_white_pixels = new Array()
        // this.dilation_black_pixels = new Array()
        this.dilation_white_pr = new PixelRecord()
        this.dilation_black_pr = new PixelRecord()
        
        // Image Models
        // stage 1 image model
        this.raw_image_model = new RawImageModel()
        // - frame models
        this.clamped_frame_model = new ClampedFrameModel()
        this.sigma_frame_model = new ThreeSigmaFrameModel()
        this.curved_frame_model = new CurvedFrameModel()
        // - mask models
        this.binary_mask_model = new BinaryMaskModel()

        // view layer
        this.image_view = new ImagePanelView(this)
        this.thumbnail_bar = new ThumbnailBar(this)
        this.image_control_panel = new ControlPanel(this)
    }

    // stage 1:
    set_raw_image_model(frame_payload, mask_payload, float_list_payload) {
        this.raw_image_model.update_frame_image(frame_payload)
        this.raw_image_model.update_mask_image(mask_payload)
        this.raw_image_model.set_zscale(float_list_payload)
    }

    // stage 1.5:
    refresh_thumbnails(coord_list, patch) {
        this.mask_image_wrapper = this.raw_image_model.mask_image
        this.thumbnail_bar.clear_thumbnails();
        this.thumbnail_bar.append_thumbnails(this.mask_image_wrapper, coord_list, patch)
        this.thumbnail_bar.refresh_thumbnails();
    }

    refresh_frame_view(start_stage) {
        switch (start_stage) {
            case FramePipeStage.RAW_STAGE:
                let [v_min, v_max] = this.raw_image_model.pixel_range
                this.image_control_panel.set_scale_clamp_range(v_min, v_max)
            case FramePipeStage.CLAMP_STAGE:
                let [clp_min, clp_max] = this.image_control_panel.get_scale_clamp_range()
                let raw_frame = this.raw_image_model.raw_frame;
                this.clamped_frame_model.update_clamped_frame(raw_frame, [clp_min, clp_max])
            case FramePipeStage.SIGMA_STAGE:
                let clamped_frame = this.clamped_frame_model.clamped_frame
                this.sigma_frame_model.update_sigma_frame(clamped_frame)
            case FramePipeStage.CURVE_STAGE:
                let curve_method_name = this.image_control_panel.get_current_scale_method_name()
                let curver = new ImageCurver(curve_method_name)
                let sigma_frame = this.sigma_frame_model.sigma_frame
                // zscale is in fact clamping
                if (curve_method_name == 'zscale') {
                    this.curved_frame_model.update_zscale_frame(sigma_frame, curver, this.raw_image_model.zscale_params)
                } else {
                    this.curved_frame_model.update_curve_frame(sigma_frame, curver)
                }
            case FramePipeStage.RENDER_STAGE:
                let [black_percent, white_percent] = this.image_control_panel.get_black_white_points_percent()
                let curved_frame = this.curved_frame_model.curved_frame
                let final_frame = lerp_color_uint8(curved_frame, black_percent, white_percent)
                // update to canvas
                let [width, height] = this.raw_image_model.image_dimension
                this.image_view.set_frame_image_pixels_from_grayscale_array(
                    width, height, final_frame.tolist())
        }

    }

    refresh_mask_view(start_stage) {
        let width = 0
        let height = 0
        let rval = []
        switch (start_stage) {
            case MaskPipeStage.RAW_STAGE:
            case MaskPipeStage.BINARY_STAGE:
                let raw_mask = this.raw_image_model.raw_mask
                let threshold = this.image_control_panel.get_threshold_value()
                this.binary_mask_model.update_binary_mask(raw_mask, threshold)
            case MaskPipeStage.RENDER_STAGE:
            case MaskPipeStage.DILATION_STAGE:
                // TODO: Need to refactor the canvas code
                let binary_mask = this.binary_mask_model.binary_mask
                rval = this.raw_image_model.image_dimension
                width = rval[0]
                height = rval[1]
                this.image_view.set_mask_image_pixels_from_grayscale_array(
                    width, height, binary_mask.tolist())
                let dilation_value = this.image_control_panel.get_dilation_value()
                if (dilation_value != 0)
                    this.image_view.post_process_mask_image_dilation(dilation_value)
            case MaskPipeStage.PIXEL_STAGE:
                rval = this.raw_image_model.image_dimension
                width = rval[0]
                height = rval[1]
                this.image_view.set_mask_image_edited_dilation_pixels(width, height,
                    this.dilation_white_pr.tolist(), this.dilation_black_pr.tolist())


        }
    }

    update_zoom_window() {
        let [width, height] = this.raw_image_model.image_dimension
        this.image_view.set_new_zoom_window_based_on_image_size(width, height)
    }

    refresh_images_and_zoom_window() {
        this.image_view.refresh_main_image_with_frame()
        this.image_view.refresh_zoomed_images()
        this.image_view.refresh_zoom_window()
    }

    refresh_zoom_images() {
        this.image_view.refresh_zoomed_images()
    }

    // Getters
    get control_panel() {
        return this.image_control_panel
    }
    
    get dilation_white_px_record() {
        return this.dilation_white_pr
    }

    get dilation_black_px_record() {
        return this.dilation_black_pr
    }
}

