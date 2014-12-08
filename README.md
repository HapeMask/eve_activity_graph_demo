Eve Activity Graph (demo)
=======================

This is a basic, barebones demonstration of a webapp using the eve_activity code. To view the killboard activity for the character "HapeMask" over the past 30 days, visit the following URL:

http://eve-activity-graph.herokuapp.com/activity/char/HapeMask/30

or, for a corporation over the past 10 days:

[http://eve-activity-graph.herokuapp.com/activity/corp/Hard Knocks Inc./10](http://eve-activity-graph.herokuapp.com/activity/corp/Hard%20Knocks%20Inc./10)

Top two peak activity times (red bars) are a slightly experimental feature and may randomly be wrong. If you're so inclined, send me the URL of any result that looks weird so I can take a look.

DISCLAIMER
===========
This is just a demo running on my personal free app host, meaning it may randomly go down. Requests for a given (name, days) pair are cached for a day, and you can't request more than 30 days. Since the zKillboard API is limited to 360 requests per hour, and 200 kills is the most per request, fetching the activity for a busy corp over any reasonable amount of time may use up many of those requests. Once the requests run out, no one can use the tool until the next hour rolls around. That's life. I'd say don't DOS me bro, but hey this *is* Eve we're talking about...


Dev-only Stuff Below!
====================

Deploying
=========
The demo comes w/all the files necessary to deploy on Heroku, or it can be run as a standalone Flask app.

To run on Heroku, you'll need to set your app to use the cedar stack (not cedar-14 since the required buildpack doesn't support it yet), and set your buildpack to this: https://github.com/thenovices/heroku-buildpack-scipy. Once you've done that, just push this repository to your heroku app remote.

To run as a standalone Flask app, make sure you have the depedencies installed, then run:

    python eve_activity_graph.py

This runs the app in debug mode (anyone can run arbitrary python code through it), so make sure you don't have port 5000 open on your machine.

Dependencies
============
* Flask
* Flask-Cache
* numpy
* scipy
* scikit-learn
