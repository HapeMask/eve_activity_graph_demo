import os
from flask import Flask
from flask import url_for, render_template, abort, request, redirect
from flask.ext.cache import Cache

from utils import get_kills_and_peaks, SECONDS_PER_DAY, ZKILL_TYPE_MAP

ACTIVITY_CACHE_TIMEOUT = SECONDS_PER_DAY
HOUR_LABELS = ["%d:00" % h for h in range(24)]

# Heroku says "read-only filesystem." Nope.
os.system("chmod -R 777 /app/static/cache")

app = Flask(__name__)
cache = Cache(app, config={"CACHE_TYPE" : "filesystem",
                           "CACHE_DIR"  : "static/cache"})

@app.route("/activity", methods=["POST", "GET"])
def activity_base():
    if request.method == "POST":
        name = request.form["name"]
        nametype = request.form["type"]
        days = request.form["days"]
        return redirect(url_for("activity",
                                name =      name,
                                nametype =  nametype,
                                days =      days))
    else:
        return render_template("activity_base.html",
                main_css_url = url_for("static", filename="css/main.css"),
                jquery_url   = url_for("static", filename="js/jquery.min.js"))

# Cache decorator needs to be first on the stack, since that's what the route
# will call.
@app.route("/activity/<nametype>/<name>/<days>")
@cache.cached(timeout=ACTIVITY_CACHE_TIMEOUT)
def activity(nametype, name, days):
    if nametype not in ZKILL_TYPE_MAP:
        abort(404)

    try:
        days = min(max(int(days), 1), 30)
    except:
        abort(404)

    try:
        bars, peaks = get_kills_and_peaks(name, nametype, days)
        success = True
        message = ""
    except Exception as e:
        bars = 24*[0]
        peaks = []
        success = False
        message = str(e.args[0])

    # We don't need to reproduce the original floats from these values, so
    # don't waste space. No one wants to read 10 decimal places on the chart or
    # in the source.
    bars = [round(b, 1) for b in bars]

    # Set a different color for the peak time bars.
    fill_colors = 24*["rgb(64,96,192)"]
    for pt in peaks:
        fill_colors[min(int(round(pt)), len(fill_colors)-1)] = "rgb(192,64,64)"

    template_vars = {"name"         : name,
                     "jquery_url"   : url_for("static", filename="js/jquery.min.js"),
                     "chart_url"    : url_for("static", filename="js/ChartNew.min.js"),
                     "main_css_url" : url_for("static", filename="css/main.css"),
                     "data"         : bars,
                     "labels"       : HOUR_LABELS,
                     "fill_colors"  : fill_colors,
                     "scale_steps"  : 1+max([int(h) for h in bars]),
                     "n_days"       : days,
                     "type"         : nametype,
                     "success"      : success,
                     "message"      : message
                     }

    return (render_template("activity.html",  **template_vars),
            200,
            {"Cache-Control": "public, max-age=%d" % ACTIVITY_CACHE_TIMEOUT})

if __name__ == "__main__":
    app.run(debug=True)
