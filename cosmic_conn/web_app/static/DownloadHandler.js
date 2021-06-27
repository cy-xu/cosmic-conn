const FILE_DOWNLOAD_SERVICE = 'download'
const DOWNLOAD_BUTTON_ID = 'file-download-btn'

class DownloadHandler {
    constructor(control_panel, status_indicator) {
        this.control_panel = control_panel
        this.status_indicator = status_indicator
        this.download_button = document.getElementById(DOWNLOAD_BUTTON_ID)
        this.init_download_button()
    }

    init_download_button() {
        $(this.download_button).click(() => {
            this.deactivate_download_button()
            let dilation = this.control_panel.get_dilation_value()      // float
            let threshold = this.control_panel.get_threshold_value()    // threshold
            let formdata = new FormData()
            formdata.append('dilation', dilation)
            formdata.append('threshold', threshold)
            let download_promise = this.download_bundled_file(formdata)
            download_promise.then(result => {
                let response = result['response']
                let filename = result['filename']
                console.log("Download filename: ", filename)
                let blob = new Blob([response])
                let a = document.createElement('a')
                a.style = "display: none"
                document.body.appendChild(a)
                let url = window.URL.createObjectURL(blob)
                a.href = url;
                a.download = filename
                a.click()
                window.URL.revokeObjectURL(url)
                this.activate_download_button()
                this.status_indicator.display_status_info("Edited binary mask is appended and saved as " + filename)
            })
        });
    }

    download_bundled_file(formdata) {
        return new Promise((resolve, reject) => {
            var xhr = new XMLHttpRequest()
            xhr.open('POST', HOST_URL + FILE_DOWNLOAD_SERVICE)
            xhr.responseType = 'blob'

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    var percent_completed = (e.loaded / e.total) * 100
                    console.log(percent_completed)
                }
            }

            xhr.onload = () => {
                console.log('status: ' + xhr.status)
                if (xhr.status == 200) {
                    let disposition = xhr.getResponseHeader('Content-Disposition')
                    let filename = "";
                    // parse attachment
                    if (disposition && disposition.indexOf('attachment') !== -1) {
                        var filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                        var matches = filenameRegex.exec(disposition);
                        if (matches != null && matches[1]) {
                            filename = matches[1].replace(/['"]/g, '');
                        }
                    }

                    let res = { "response": xhr.response, "filename": filename }
                    resolve(res)
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

    activate_download_button() {
        this.download_button.disabled = false
    }

    deactivate_download_button() {
        this.download_button.disabled = true
    }
}