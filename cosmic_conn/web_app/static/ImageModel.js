class ImageDataWrapper {
    constructor(grayscale_array, width, height) {
        this.image_array = nj.float32(grayscale_array)
        this.width = width
        this.height = height
        this.pixel_min = nj.min(this.image_array)
        this.pixel_max = nj.max(this.image_array)
    }
}

class RawImageModel {
    // stage 1 data model
    constructor() {
        this.frame_image = null
        this.mask_image = null
        this.pixel_value_range = [0, 0]
    }

    update_frame_image(frame_payload) {
        let width = frame_payload.image_width
        let height = frame_payload.image_height
        let grayscale_array = frame_payload.image_grayscale_array
        this.frame_image = new ImageDataWrapper(grayscale_array, width, height)
    }

    update_mask_image(mask_payload) {
        let width = mask_payload.image_width
        let height = mask_payload.image_height
        let mask_array = mask_payload.image_grayscale_array
        this.mask_image = new ImageDataWrapper(mask_array, width, height)
    }

    gray_value_at(x, y) {
        let w = this.frame_image.width
        let index = w * Math.floor(y) + Math.floor(x)
        return this.frame_image.image_array.get(index)
    }

    mask_value_at(x, y) {
        let w = this.frame_image.width
        let index = w * Math.floor(y) + Math.floor(x)
        return this.mask_image.image_array.get(index)
    }

    set_zscale(float_list_payload) {
        let zscale_params = float_list_payload.zscale_array
        let zscale_z1 = zscale_params[0]
        let zscale_z2 = zscale_params[1]
        this.zscale_params = [zscale_z1, zscale_z2]
    }

    get image_dimension() {
        return [this.frame_image.width, this.frame_image.height]
    }

    get pixel_range() {
        return [this.frame_image.pixel_min, this.frame_image.pixel_max]
    }

    get raw_frame() {
        return this.frame_image.image_array.clone()
    }

    get raw_mask() {
        return this.mask_image.image_array.clone()
    }
}

class ClampedFrameModel {
    // stage 2 data model
    constructor() {
        this.clamped_frame_array = null
        this.pixel_min = 0
        this.pixel_max = 0
    }

    update_clamped_frame(raw_frame_array, clamp_range) {
        let [min, max] = clamp_range
        this.pixel_min = min
        this.pixel_max = max
        this.clamped_frame_array = nj.clip(raw_frame_array, min, max)
    }

    get clamped_frame() {
        return this.clamped_frame_array.clone()
    }
}

class ThreeSigmaFrameModel {
    // stage 3 data model
    constructor() {
        this.sigma_frame_array = null
        this.low_bound = 0
        this.high_bound = 0
    }

    update_sigma_frame(frame_array) {
        let med = nj.mean(frame_array)
        let std = nj.std(frame_array)
        this.low_bound = med - 3 * std
        this.high_bound = med + 3 * std
        this.sigma_frame_array = nj.clip(frame_array, this.low_bound, this.high_bound)
    }

    get sigma_frame() {
        return this.sigma_frame_array.clone()
    }

    get value_range() {
        return [this.low_bound, this.high_bound]
    }
}

class CurvedFrameModel {
    // stage 4 data model
    constructor() {
        this.curved_frame_array = null
        this.pixel_min = 0
        this.pixel_max = 0
    }

    update_curve_frame(frame_array, curver) {
        this.curved_frame_array = curver.curve_image_array(frame_array)
        this.pixel_min = nj.min(this.curved_frame_array)
        this.pixel_max = nj.max(this.curved_frame_array)
    }

    update_zscale_frame(frame_array, curver, zscale_params) {
        this.curved_frame_array = curver.zscale_image_array(frame_array, zscale_params)
        this.pixel_min = nj.min(this.curved_frame_array)
        this.pixel_max = nj.max(this.curved_frame_array)
    }

    get curved_frame() {
        return this.curved_frame_array.clone()
    }

    get pixel_range() {
        return [this.pixel_min, this.pixel_max]
    }
}

class BinaryMaskModel {
    constructor() {
        this.binary_mask_array = null
    }

    update_binary_mask(mask_array, threshold) {
        let binary_mask_array = mask_array.clone()
        let length = binary_mask_array.shape[0]
        for (let i = 0; i < length; i++) {
            let val = binary_mask_array.get(i) > threshold ? 255 : 0
            binary_mask_array.set(i, val)
        }
        this.binary_mask_array = binary_mask_array
    }

    get binary_mask() {
        return this.binary_mask_array
    }
}

class Pixel {
    constructor(x, y) {
        this.x_coord = x
        this.y_coord = y
    }
    
    in_range(width, height) {
        let x_in_range = this.x_coord >= 0 && this.x_coord < width;
        let y_in_range = this.y_coord >= 0 && this.y_coord < height;
        return x_in_range && y_in_range
    }

    get x() {return this.x_coord}
    get y() {return this.y_coord}
}
