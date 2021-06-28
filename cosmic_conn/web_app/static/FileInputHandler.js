// IDs should match those used in the main html
const FILE_INPUT_ID = 'file-input'
const FILE_NAME_LABEL_ID = 'file-name-label'
const FILE_UPLOAD_BUTTON_ID = 'file-upload-btn'

const FILE_UPLOAD_SERVICE = 'process'
const FILE_UPLOAD_RESPONSE_TYPE = 'arraybuffer'

const FILE_ID_IN_FORMDATA = 'file'

class FileInputHanlder {
    constructor(status_indicator) {
        this.status_indicator = status_indicator
        this.file_input = document.getElementById(FILE_INPUT_ID)
        this.file_name_label = document.getElementById(FILE_NAME_LABEL_ID)
        this.file_upload_button = document.getElementById(FILE_UPLOAD_BUTTON_ID)
    }

    init_file_input_listener() {
        $('#' + FILE_INPUT_ID).change(() => {
            let selected_file = this.file_input.files[0];
            this.file_name_label.innerHTML = selected_file.name;
        });
    }

    // file_upload_callback needs to have 
    init_upload_button_click_behavior(file_upload_response_callback) {
        $('#' + FILE_UPLOAD_BUTTON_ID).click(() => {
            if (this.file_input.files.length < 1) {
                alert('You need to select a file first!')
            }
            let formdata = new FormData()
            // 1. first step attach the file
            formdata = this.wrap_file_into_formdata(formdata)
            // 2. second step attach the uuid
            formdata = this.wrap_uuid_into_formdata(formdata, uuid)
            let transfer_promise = this.transfer_selected_file(formdata)
            transfer_promise.then(buffer => {
                file_upload_response_callback(buffer)
            }, error_message => {
                this.status_indicator.display_status_error(error_message)
            })
        })
    }

    wrap_file_into_formdata(formdata) {
        let selected_file = this.file_input.files[0]
        formdata.append(FILE_ID_IN_FORMDATA, selected_file)
        return formdata
    }

    wrap_uuid_into_formdata(formdata) {
        formdata.append("uuid", uuid)
        return formdata
    }

    transfer_selected_file(formdata) {
        return new Promise((resolve, reject) => {
            var xhr = new XMLHttpRequest()
            xhr.open('POST', HOST_URL + FILE_UPLOAD_SERVICE)
            xhr.responseType = FILE_UPLOAD_RESPONSE_TYPE

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

            xhr.send(formdata)
        })
    }
}