import io
import sys
import json
import zlib
import datetime
import xml.etree.ElementTree as xmlet

if sys.version_info.major == 3:
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    status = lambda resp : resp.status
else:
    from urllib2 import urlopen, Request, quote
    status = lambda resp : resp.getcode()

from eve_activity import compute_peak_times, compute_avg_kills_by_hour

EPOCH_START = datetime.date(1970,1,1).toordinal()
SECONDS_PER_DAY = 24*60*60

EVE_API_CHAR_ID_URL = "https://api.eveonline.com/eve/CharacterID.xml.aspx?names=%s"

ZKILL_API_BASE = "https://zkillboard.com/api"
ZKILL_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ZKILL_TIMESTAMP_FORMAT = "%Y%m%d%H%M"
ZKILL_MAX_KILLS_PER_RESPONSE = 200

# Allows zlib.decompress to decompress gzip-compressed strings as well.
# From zlib.h header file, not documented in Python.
ZLIB_DECODE_AUTO = 32 + zlib.MAX_WBITS

def decompress(s):
    """Decode a gzip compressed string."""
    return zlib.decompress(s, ZLIB_DECODE_AUTO)

def date_to_timestamp(date):
    """Convert a datetime.date object to a seconds-since-epoch timestamp."""
    return (date.toordinal() - EPOCH_START) * SECONDS_PER_DAY

def character_id_from_name(name):
    """Get a character/corporation ID by name using the Eve API."""
    response = urlopen(EVE_API_CHAR_ID_URL % quote(name))
    if status(response) not in [200,304]:
        raise RuntimeError("Got invalid HTTP status from Eve API: %d" % status(response))
    tree = xmlet.fromstring(response.read())

    cid = int(tree.find("result/rowset/row").get("characterID"))
    if cid == 0:
        raise ValueError("ID for %s not found." % name)
    return cid

def make_zkill_api_url(cid, id_type, n_days, page):
    now = datetime.datetime.now()
    end_ts = now.strftime(ZKILL_TIMESTAMP_FORMAT)
    start_ts = (now - datetime.timedelta(days=n_days)).strftime(ZKILL_TIMESTAMP_FORMAT)

    return "/".join([ZKILL_API_BASE,
                     id_type, str(cid),
                     "startTime", start_ts, "endTime", end_ts,
                     "orderDirection", "desc",
                     "page", str(page)])

def kill_times_from_zkill_json(zkill_json):
    """Extract date-only timestamps and fractional hour-of-the-day values from
    zkillboard-sourced kill information (JSON).

    Parameters
    ----------
    zkill_json : list of dicts
        A JSON object dump from zkillboard's API containing a list of kills of
        interest.

    Returns
    -------
    (timestamps, hours) : (list of int, list of float)
        Date-only timestamps and fractional hour-of-the-day values for each
        kill in the given JSON.
    """

    times = [k["killTime"] for k in zkill_json]
    datetimes = [datetime.datetime.strptime(time, ZKILL_DATETIME_FORMAT) for time in times]

    # Convert just the date to timestamp form (ignoring hours/mins/seconds) so
    # that it can be use to uniquely identify the day on which each kill
    # occurred.
    return zip(*[(date_to_timestamp(dt.date()),
                 (dt.hour + dt.minute/60. + dt.second/(60.**2)))
                for dt in datetimes])

def get_kills_for_id(cid, id_type, n_days):
    """Gets a list of JSON-decoded kills from zkillboard for the given
    character ID over the past n_days days."""
    page = 1
    last_result_len = ZKILL_MAX_KILLS_PER_RESPONSE
    kills = []

    while last_result_len == ZKILL_MAX_KILLS_PER_RESPONSE:
        try:
            zkill_url = make_zkill_api_url(cid, id_type, n_days, page)
            request = Request(zkill_url, headers = {
                "Accept-Encoding" : "gzip",
                "User-Agent" : "Eve Activity Graph Demo, Maintainer: hapemask@gmail.com"
                })
            response = urlopen(request)
            if status(response) not in [200,304]:
                raise RuntimeError("Got invalid HTTP status from ZKill: %d" % status(response))

            if response.info().get("Content-Encoding") == "gzip":
                data = decompress(response.read())
            else:
                data = response.read()

            if isinstance(data, bytes):
                data = data.decode(encoding="utf-8")

            kills_chunk = json.loads(data)
            kills.extend(kills_chunk)
            last_result_len = len(kills_chunk)
            page += 1
        except:
            break

    return kills

def get_kills_for_char(name, n_days):
    cid = character_id_from_name(name)
    return get_kills_for_id(cid, "characterID", n_days)

def get_kills_for_corp(name, n_days):
    cid = character_id_from_name(name)
    return get_kills_for_id(cid, "corporationID", n_days)

def get_kills_and_peaks(name, name_type, n_days, n_peaks=2):
    if name_type == "char":
        kills = get_kills_for_char(name, n_days)
    elif name_type == "corp":
        kills = get_kills_for_corp(name, n_days)
    else:
        raise ValueError("Invalid name type: %s" % name_type)

    if len(kills) == 0:
        return 24*[0], n_peaks*[0]

    date_timestamps, hours = kill_times_from_zkill_json(kills)

    avg_kills_by_hour = compute_avg_kills_by_hour(date_timestamps, hours)
    peak_times = compute_peak_times(hours, n_peaks)

    return avg_kills_by_hour, peak_times
