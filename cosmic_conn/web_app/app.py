# -*- coding: utf-8 -*-

"""
Main file for the Cosmic-CoNN web-based CR detector.
Boning Dong, CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np
from astropy.io import fits
from skimage.morphology import dilation, square

try:
    from flask_apscheduler import APScheduler
    from flask import (
        Flask,
        request,
        render_template,
        send_from_directory,
        send_file,
        make_response,
        url_for,
        jsonify,
        abort,
        session,
    )
except ImportError:
    raise ImportError(
        "Flask is not installed.\n"
        "Please run `pip install cosmic-conn[webapp]` to install packages for the web app."
    )

from cosmic_conn.web_app.toolkit.astromsg import (
    FloatListPayload,
    ImagePayload,
    PostResponse,
    ThumbnailPayload,
)
from cosmic_conn.web_app.toolkit.file_manager import UploadFileManager, file_recycle_thread
from cosmic_conn.web_app.toolkit.toolkit import find_noise_thumbnail_coords

# from templates.inference import CR_detector
from cosmic_conn.cr_pipeline.utils_img import zscale

# debug
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"fits", "fz"}

app = Flask(__name__)

# app.config["EXPLAIN_TEMPLATE_LOADING"] = True
app.secret_key = "local_instance"

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Upload Storage
FILE_RECYCLE_JOB_ID = "file-recycle-id"
FILE_RECYCLE_JOB_INTERVAL = 10

# For local instance, uploads stored at current working directory
# STORAGE_PATH = os.path.join(app.instance_path, "upload_storage")
STORAGE_PATH = os.path.join(os.getcwd(), "instance_temp_storage")
OUTPUT_PATH = os.path.join(os.getcwd(), "cosmic_conn_output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

file_manager = UploadFileManager(STORAGE_PATH)
# scheduler.add_job(
#     id=FILE_RECYCLE_JOB_ID,
#     func=file_recycle_thread,
#     args=[file_manager],
#     trigger='interval',
#     seconds=FILE_RECYCLE_JOB_INTERVAL
# )

# Thumbnail Numbers
THUMBNAIL_PATCH_SIZE = 64
THUMBNAIL_NUMBER = 20


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/process", methods=["POST"])
def process():
    # get user ip address and uuid address
    try:
        user_ip = request.remote_addr
        user_uuid = request.form["uuid"]
        print("user ip: ", user_ip, "user_id: ", user_uuid)
    except:
        abort(400, "Cannot parse user ip or user id.")

    original_file = request.files["file"]

    # https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
    if not original_file or not allowed_file(original_file.filename):
        return abort(400, "Only *.fits or *.fz files are allowed.")

    # create file key based on user's ip  and uuid
    file_key = "{}_{}".format(user_ip, user_uuid).replace(".", "-")
    file_manager.store_uploaded_file(file_key, original_file)
    status, fits_file_path = file_manager.fetch_uploaded_file_path(file_key)
    # print("fetched file path: ", fits_file_path)
    if not status:
        return abort(400, "Save failed or cannot fetch saved image.")

    # model and settings
    opt = app.config['opt']
    cr_model = app.config['cr_model']

    # start detection:
    with fits.open(fits_file_path) as hdul:
        image = None
        try:
            if opt.ext != 'SCI':
                # if user asigned extension, try it first
                image = hdul[opt.ext].data.astype("float32")

            elif hdul[0].data is not None:
                image = hdul[0].data.astype("float32")

            elif hdul[1].data is not None:
                image = hdul[1].data.astype("float32")

            else:
                image = hdul[opt.ext].data.astype("float32")

            # remove Nan is exist
            image = np.nan_to_num(image, copy=False, nan=0.0)

        except:
            raise ValueError(
                f"No valid data found in extention 0, 1 or {opt.ext}, \
                    -e to specify extension name.")

        # CR detection
        cr_probability = cr_model.detect_cr(image, ret_numpy=True)

        # append the CR probability mask and save a copy
        hdul.append(
            fits.CompImageHDU(cr_probability, name="CR_probability")
        )

        # save a FITS copy with CR probability mask
        fits_out_name = "CR_" + original_file.filename
        fits_out_path = os.path.join(OUTPUT_PATH, fits_out_name)
        hdul.writeto(fits_out_path, overwrite=True)

    session["fits_out_path"] = fits_out_path
    session["fits_out_name"] = fits_out_name

    # save the CR array for later
    cr_mask_path = os.path.join(
        STORAGE_PATH, original_file.filename+"_cr.npy")
    np.save(cr_mask_path, cr_probability)
    session["cr_mask_path"] = cr_mask_path

    # flip for correct orientation before drawing
    image = np.flipud(image)
    cr_probability = np.flipud(cr_probability)

    # Construct response to send back
    post_response = PostResponse()
    # Append Image Payload
    post_response.append_payload(ImagePayload(image))
    post_response.append_payload(ImagePayload(cr_probability))

    # Append Thumbnail
    patch = THUMBNAIL_PATCH_SIZE
    binary_threshold = 0.5
    thumbnail_coords = find_noise_thumbnail_coords(
        cr_probability, binary_threshold, patch
    )
    thumbnail_number = min(THUMBNAIL_NUMBER, len(thumbnail_coords))
    thumbnail_payload = ThumbnailPayload(patch)
    for coord in thumbnail_coords[0:thumbnail_number]:
        thumbnail_payload.add_thumbnail_coord(coord)
    post_response.append_payload(thumbnail_payload)

    # Append zscale data
    zscale_list = list(zscale(image))
    post_response.append_payload(FloatListPayload(zscale_list))

    response = make_response(post_response.tobytes(), 200)
    response.headers.set("Content-Type", "application/octet-stream")

    return response


@app.route("/detect_gpu", methods=["POST"])
def detect_gpu():
    # gpu could have different number but cpu is consistent
    gpu_not_found = str(app.config['cr_model'].device) == 'cpu'
    # 1. call gpu function
    # 2. json obj {"gpu_found": true}
    # 3. return response
    response = {'gpu_detected': not gpu_not_found}
    return jsonify(response)


@app.route("/download", methods=["POST"])
def download():
    # when user download, takes two parameters from the front end,
    # process the mask append it after the CR probability mask
    try:
        n_dilation = int(request.form["dilation"])
        threshold = float(request.form["threshold"])
        print("[download] n_dilation: ", n_dilation, "threshold: ", threshold)
    except:
        abort(400, "Cannot parse dilation or threshold.")

    cr_probability = np.load(session["cr_mask_path"], allow_pickle=True)
    # 1. threshold
    cr_binary = cr_probability > threshold
    # 2. dilation
    for i in range(0, n_dilation):
        cr_binary = dilation(cr_binary, square(3))

    with fits.open(session["fits_out_path"]) as hdul:
        cr_binary = cr_binary.astype("uint8")
        hdul.append(fits.CompImageHDU(
            cr_binary, name="CR_mask", uint=True)
        )
        hdul.writeto(session["fits_out_path"], overwrite=True)

    # then show a message "Binary mask appended to a new FITS copy and saved
    #  in "cosmic_conn_output"."
    return send_file(
        session["fits_out_path"], as_attachment=True,
        attachment_filename=session["fits_out_name"]
    )


@app.route("/fitstest", methods=["GET"])
def fitstest():
    print("received fitstest get")

    return make_response(url_for("static", filename="test-fits.fz"), 200)
    # return make_response("", 200)


@app.route("/get_file/<file>")
def get_file(file):
    return send_from_directory(app.config["session_root"], file)


def main(cr_model, opt):
    # # init model based on console arguments
    app.config['opt'] = opt
    app.config['cr_model'] = cr_model

    app.run(host="127.0.0.1", debug=False)


if __name__ == "__main__":
    # from requests import get
    # ip = get("https://api.ipify.org").text

    main()
