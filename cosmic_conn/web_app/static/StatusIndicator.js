const STATUS_INDICATOR_ID = 'status-indicator'
const STATUS_INDICATOR_DISPLAY_DURATION = 10000 //ms

class StatusIndicator {
    constructor() {
        this.timeout = null;
        this.status_bar = document.getElementById(STATUS_INDICATOR_ID)
        this.status_bar_jquery = $('#' + STATUS_INDICATOR_ID)
        this.hide_status_bar()
    }

    hide_status_bar() {
        this.status_bar_jquery.hide();
    }

    set_info_color() {
        this.status_bar_jquery.removeClass('alert-danger')
        this.status_bar_jquery.addClass('alert-info')
    }

    set_error_color() {
        this.status_bar_jquery.removeClass('alert-info')
        this.status_bar_jquery.addClass('alert-danger')
    }

    clear_current_timeout() {
        if (this.timeout == null)
            return;
        clearTimeout(this.timeout)
    }

    display_status_info(info) {
        this.clear_current_timeout()
        this.set_info_color()
        this.status_bar_jquery.show()
        this.status_bar_jquery.text('Info: ' + info)
        // this.timeout = setTimeout(() => {
        //     this.status_bar_jquery.fadeOut('slow')
        //     this.timeout = null;
        // }, STATUS_INDICATOR_DISPLAY_DURATION);
    }

    display_status_error(error) {
        this.clear_current_timeout()
        this.set_error_color()
        this.status_bar_jquery.show()
        this.status_bar_jquery.text('Error: ' + error)
        // this.timeout = setTimeout(() => {
        //     this.status_bar_jquery.fadeOut('slow')
        //     this.timeout = null;
        // }, STATUS_INDICATOR_DISPLAY_DURATION);
    }

}