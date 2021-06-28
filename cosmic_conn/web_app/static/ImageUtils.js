const FLOAT_MAXVALUE = 3.40282347e+38; // largest positive number in float32
const FLOAT_MINVALUE = -3.40282347e+38; // largest negative number in float32


// note: several things to add a new scale method
// 1. Add corresponding name in FrameControls.html
// 2. Create a const here indicate that name in lower case
// 3. Use the name in scale_and_clamp_grayscale_array method in this file
const LINEAR_SCALE_NAME = 'linear'
const ZSCALE_NAME = 'zscale'
const LOG_SCALE_NAME = 'log'
const SQUARE_ROOT_SCALE_NAME = 'sqrt'


function lerp_color_uint8(image_array, bottom_percent, top_percent) {
    let num_array = nj.float32(image_array.clone())
    let [min_value, max_value] = [nj.min(num_array), nj.max(num_array)]
    let dark_val = (max_value - min_value) * bottom_percent + min_value
    let bright_val = (max_value - min_value) * top_percent + min_value

    let val_range = bright_val - dark_val
    for (let i = 0; i < num_array.size; i++) {
        let grayscale_offset = num_array.get(i) - dark_val;
        if (grayscale_offset < 0) {
            num_array.set(i, 0)
        } else if (grayscale_offset > val_range) {
            num_array.set(i, 255)
        } else {
            num_array.set(i, Math.round(grayscale_offset / val_range * 255))
        }
    }
    return num_array
}

class ImageCurver {
    constructor(method_name) {
        this.method_name = method_name;
    }

    zscale_image_array(image_array, zscale_params) {
        let zscale_z1 = zscale_params[0]
        let zscale_z2 = zscale_params[1]

        image_array = nj.clip(image_array, zscale_z1, zscale_z2)
        return image_array
    }

    curve_image_array(image_array) {
        switch (this.method_name) {
            case LINEAR_SCALE_NAME:
                return this.#linear_curve(image_array)
            case LOG_SCALE_NAME:
                return this.#log_curve(image_array)
            case SQUARE_ROOT_SCALE_NAME:
                return this.#sqrt_curve(image_array)
            default:
                console.log('Error: invalid method name: ', this.method_name)
        }
    }

    #linear_curve(image_array) {
        return image_array.clone()
    }

    #log_curve(image_array) {
        let min_val = nj.min(image_array)
        if (min_val < 1) {
            let offset = 1 - min_val
            image_array = nj.add(image_array, offset)
        }

        return nj.log(image_array)
    }

    #sqrt_curve(image_array) {
        let min_val = nj.min(image_array)
        if (min_val < 0) {
            let offset = 0 - min_val
            image_array = nj.add(image_array, offset)
        }

        return nj.sqrt(image_array)
    }
}

class FramImageProcessor {
    constructor(method_name) {
        this.clamp_min = FLOAT_MINVALUE
        this.clamp_max = FLOAT_MAXVALUE
        this.method_name = method_name
    }

    set_clamp_range(min, max) {
        if (max < min)
            return
        this.clamp_min = min
        this.clamp_max = max
    }

    // 1st: scale based on the map method and the clamp value
    scale_and_clamp_grayscale_array(grayscale_array) {
        switch (this.method_name) {
            case LINEAR_SCALE_NAME:
                return this.#linear_scale(grayscale_array)
            case ZSCALE_SCALE_NAME:
                return this.#zscale(grayscale_array)
            case LOG_SCALE_NAME:
                return this.#log_scale(grayscale_array)
            case SQUARE_ROOT_SCALE_NAME:
                return this.#sqrt_scale(grayscale_array)
            default:
                console.log('Error: invalid method name: ', this.method_name)
        }
    }

    #linear_scale(grayscale_array) {
        const pixel_number = grayscale_array.length
        for (let i = 0; i < pixel_number; i++) {
            let val = grayscale_array[i]
            grayscale_array[i] = this.#clamp_value(val)
        }
        return grayscale_array
    }

    #zscale(grayscale_array) {
        const pixel_number = grayscale_array.length
        for (let i = 0; i < pixel_number; i++) {
            let val = grayscale_array[i]
            grayscale_array[i] = this.#clamp_value(val)
        }
        return grayscale_array
    }

    #log_scale(grayscale_array) {
        let num_array = nj.float32(grayscale_array)
        grayscale_array = nj.clip(num_array, this.clamp_min, this.clamp_max)

        let min_in_array = nj.min(grayscale_array)
        if (min_in_array < 1) {
            let offset = 1 - min_in_array
            grayscale_array = nj.add(grayscale_array, offset)
        }

        grayscale_array = nj.log(grayscale_array)
        return grayscale_array.tolist()
    }

    #sqrt_scale(grayscale_array) {
        let num_array = nj.float32(grayscale_array)
        grayscale_array = nj.clip(num_array, this.clamp_min, this.clamp_max)

        let min_in_array = nj.min(grayscale_array)
        if (min_in_array < 1) {
            let offset = 0 - min_in_array
            grayscale_array = nj.add(grayscale_array, offset)
        }

        grayscale_array = nj.sqrt(grayscale_array)
        return grayscale_array.tolist()
    }

    #clamp_value(val) {
        if (val < this.clamp_min)
            return this.clamp_min
        if (val > this.clamp_max)
            return this.clamp_max
        return val
    }

    // 2nd: lerp the color
    lerp_color_to_uint8(grayscale_array, bottom_percent, top_percent) {
        // let [dark_val, bright_val] = this.#get_dark_bright_values(bottom_percent, top_percent)
        let num_array = nj.float32(grayscale_array)
        let [min_value, max_value] = [nj.min(num_array), nj.max(num_array)]
        let dark_val = (max_value - min_value) * bottom_percent + min_value
        let bright_val = (max_value - min_value) * top_percent + min_value

        let val_range = bright_val - dark_val
        for (let i = 0; i < num_array.size; i++) {
            let grayscale_offset = num_array.get(i) - dark_val;
            if (grayscale_offset < 0) {
                num_array.set(i, 0)
            } else if (grayscale_offset > val_range) {
                num_array.set(i, 255)
            } else {
                num_array.set(i, Math.round(grayscale_offset / val_range * 255))
            }
        }
        return num_array.tolist()
    }
}