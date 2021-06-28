/**
 * This file is used for encapsulating the received buffer
 * This file should be synchronized with astromsg.py file
 */


const PayloadTypes = {
    NONE_TYPE: 0,
    IMAGE_TYPE: 1,
    THUMBNAIL_TYPE: 2,
    FLOATLIST_TYPE: 3
}

class PostResponse {
    // PostResponse format:
    // [Payload Count: uint32][ payload 1 ][ payload 2 ]
    // Payload format:
    // [payload size: uint32][payload type: uint8][payload]
    constructor(buffer) {
        this.response_buffer = buffer
        this.payload_counts = this.parse_payload_counts(buffer)
        this.payloads = this.#parse_all_payloads_from_buffer(buffer)
    }

    parse_payload_counts(response_buffer) {
        const response_meta_bytes = Uint32Array.BYTES_PER_ELEMENT * 1
        const repsonse_meta_offset = 0
        let response_meta_buffer =
            response_buffer.slice(repsonse_meta_offset, repsonse_meta_offset + response_meta_bytes)
        let response_meta_array = new Uint32Array(response_meta_buffer)
        return response_meta_array[0]
    }

    #parse_all_payloads_from_buffer(response_buffer) {
        let payloads = []
        const payloads_area_offset = Uint32Array.BYTES_PER_ELEMENT * 1
        let current_offset = payloads_area_offset
        while (current_offset < response_buffer.byteLength) {
            const payloads_buffer = response_buffer.slice(current_offset)
            const [payload, payload_type, payload_and_meta_bytes] =
                this.#parse_first_payload_and_meta_from_buffer(payloads_buffer)
            payloads.push(payload)
            current_offset += payload_and_meta_bytes
        }
        return payloads
    }

    #parse_first_payload_and_meta_from_buffer(payloads_buffer) {
        const payload_meta_bytes = Uint32Array.BYTES_PER_ELEMENT * 1 + Uint8Array.BYTES_PER_ELEMENT * 1
        const payload_meta_offset = 0
        let payload_meta_buffer = payloads_buffer.slice(payload_meta_offset, payload_meta_offset + payload_meta_bytes)
        let payload_size_buffer = payload_meta_buffer.slice(0, 0 + Uint32Array.BYTES_PER_ELEMENT)
        let payload_type_buffer = payload_meta_buffer.slice(Uint32Array.BYTES_PER_ELEMENT)
        let payload_bytes = (new Uint32Array(payload_size_buffer))[0];
        let payload_type = (new Uint8Array(payload_type_buffer))[0];

        const payload_offset = payload_meta_bytes
        let payload_buffer = payloads_buffer.slice(payload_offset, payload_offset + payload_bytes)
        let payload = this.#create_payload_object_based_on_type(payload_buffer, payload_type);
        let payload_and_meta_bytes = payload_bytes + payload_meta_bytes
        return [payload, payload_type, payload_and_meta_bytes]
    }

    #create_payload_object_based_on_type(payload_buffer, type) {
        switch (type) {
            case PayloadTypes.IMAGE_TYPE:
                return new ImagePayload(payload_buffer)
            case PayloadTypes.THUMBNAIL_TYPE:
                return new ThumbnailPayload(payload_buffer)
            case PayloadTypes.FLOATLIST_TYPE:
                return new FloatListPayload(payload_buffer)
            default:
                console.log("received invalid payload type")
        }
    }

    get_payload_counts() {
        return this.payload_counts
    }

    get_payload_at(index) {
        return this.payloads[index]
    }
}

class ImagePayload {
    // ImagePayload has the following memory format
    // [width: uint32][height: uint32][ image pixels: float32 array]
    constructor(buffer) {
        this.payload_buffer = buffer
        const [width, height] = this.parse_image_dimension(buffer)
        this.width = width
        this.height = height
        this.image_buffer = this.parse_image_buffer(buffer)
    }

    parse_image_dimension(payload_buffer) {
        const image_meta_bytes = Uint32Array.BYTES_PER_ELEMENT * 2
        const image_meta_offset = 0
        let image_meta_buffer =
            payload_buffer.slice(image_meta_offset, image_meta_offset + image_meta_bytes)
        let image_meta_array = new Uint32Array(image_meta_buffer)
        return [image_meta_array[0], image_meta_array[1]]
    }

    parse_image_buffer(payload_buffer) {
        const image_buffer_offset = Uint32Array.BYTES_PER_ELEMENT * 2
        return payload_buffer.slice(image_buffer_offset)
    }

    get image_width() {
        return this.width
    }

    get image_height() {
        return this.height
    }

    get image_grayscale_array() {
        return new Float32Array(this.image_buffer)
    }
}

class ThumbnailPayload {
    // Thumnail Payload has the following memory format
    // [thumbnail number: uint32][patch size: uint32][coordinate array: [x:uint32, y:uint32, x:uint32, y:uint32 ... ]
    constructor(buffer) {
        this.payload_buffer = buffer
        const [thumnail_number, patch_size] = this.parse_thumnail_meta(this.payload_buffer)
        this.thumnail_number = thumnail_number
        this.patch_size = patch_size
        this.coords_array = this.parse_thumnail_coords(this.payload_buffer)
    }

    parse_thumnail_meta(payload_buffer) {
        const thumnail_meta_bytes = Uint32Array.BYTES_PER_ELEMENT * 2
        const thumnail_meta_offset = 0
        const thumnail_meta_array = new Uint32Array(payload_buffer.slice(thumnail_meta_offset, thumnail_meta_offset + thumnail_meta_bytes))
        return [thumnail_meta_array[0], thumnail_meta_array[1]]
    }

    parse_thumnail_coords(payload_buffer) {
        const offset = Uint32Array.BYTES_PER_ELEMENT * 2
        const coord_raw_array = new Uint32Array(payload_buffer.slice(offset))
        let coords_array = []
        for (let i = 0; i < coord_raw_array.length; i = i + 2) {
            coords_array.push([coord_raw_array[i], coord_raw_array[i + 1]])
        }
        return coords_array
    }

    get thumbnail_number() {
        return this.thumnail_number
    }

    get thumbnail_patch_size() {
        return this.patch_size
    }

    get thumbnail_coords_array() {
        return this.coords_array
    }

}

class FloatListPayload {
    // FloatListPayload has the following memory format
    // [length of list: uint32][list of data in float32: [zscale z1, zscale z2, ...]]
    constructor(buffer) {
        this.payload_buffer = buffer
        this.zscale_params = new Float32Array(this.parse_zscale_buffer(buffer))
    }

    parse_zscale_buffer(payload_buffer) {
        const zscale_payload_offset = Float32Array.BYTES_PER_ELEMENT * 2
        const offset = 0
        return payload_buffer.slice(offset, offset + zscale_payload_offset)
    }

    get zscale_array() {
        // correct here
        return this.zscale_params
    }
}

// error: TypeError: Cannot set property payload_counts of #<PostResponse> which has only a getter
