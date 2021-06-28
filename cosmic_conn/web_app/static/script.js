host_url = HOST_URL;
uuid = createUUID();
/**
 * 1. send fits / return fits <-
 * 2. send array / return array <- base64
 * 
 */

var status_indicator = null
var file_input_hanlder = null
var file_download_handler = null
var controller = null


$(document).ready(function () {
    status_indicator = new StatusIndicator();
    file_input_hanlder = new FileInputHanlder(status_indicator)

    controller = new ImageController()
    file_download_handler = new DownloadHandler(controller.image_control_panel, status_indicator)

    file_input_hanlder.init_file_input_listener()
    const status_string = 'GPU not found, detection will take longer.'
    status_indicator.display_status_info(status_string)

    file_input_hanlder.init_upload_button_click_behavior(file_upload_response_callback)
    check_gpu_info()
    disableScroll()
})

function check_gpu_info() {
    let promise = new Promise((resolve, reject) => {
        var xhr = new XMLHttpRequest()
        xhr.open('POST', HOST_URL + 'detect_gpu')
        xhr.responseType = 'json'

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                var percent_completed = (e.loaded / e.total) * 100
            }
        }

        xhr.onload = () => {
            if (xhr.status == 200) {
                resolve(xhr.response)
            } else {
                const error_message =
                    `the upload failed with status: ${xhr.status} message: ${xhr.statusText}`
                reject(error_message)
            }
        }

        xhr.onerror = () => reject('error occurred.')

        xhr.send()
    })
    promise.then(response => {
        try {
            if (response.gpu_detected)
                status_indicator.display_status_info("GPU found!")
            else
                status_indicator.display_status_info("GPU not found, detection will take longer.")
        }
        catch {
            status_indicator.display_status_info("Got invalid GPU status!")
        }
    })
}

function file_upload_response_callback(arraybuffer) {
    // const status_string = 'succeed! received data buffer size: ' + arraybuffer.byteLength
    const status_string = 'Detection completed. Result saved in `cosmic_conn_output`. Zoom in on image for close inspection.'
    status_indicator.display_status_info(status_string)
    let response = new PostResponse(arraybuffer)

    // read the frame and mask image
    let frame_payload = response.get_payload_at(0)
    let mask_payload = response.get_payload_at(1)

    // read the zscale parameters
    let float_list_payload = response.get_payload_at(3)

    // get thumbnail
    let thumbnail_payload = response.get_payload_at(2)
    let patch_size = thumbnail_payload.thumbnail_patch_size
    let thumbnail_number = thumbnail_payload.thumbnail_number
    let thumbnail_coords = thumbnail_payload.thumbnail_coords_array

    controller.set_raw_image_model(frame_payload, mask_payload, float_list_payload)
    controller.refresh_thumbnails(thumbnail_coords, patch_size)
    controller.refresh_frame_view(FramePipeStage.RAW_STAGE)
    controller.refresh_mask_view(MaskPipeStage.RAW_STAGE)
    controller.update_zoom_window()
    controller.refresh_images_and_zoom_window()

    file_download_handler.activate_download_button()
}