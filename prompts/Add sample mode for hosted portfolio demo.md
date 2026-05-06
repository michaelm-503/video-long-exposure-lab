# Codex Prompt 11 — Add sample mode for hosted portfolio demo

Add a sample/demo mode so the hosted app works even without user uploads.

Files to modify:
- app.py
- sample_data/README.md
- README.md

Functional requirements:
1. Add a button for demo gallery in the select images bar.
2. Create a gallery view of reference frames pulling from the following list (a mix of personal icloud links and videos on flickr with CC license):
    - https://share.icloud.com/photos/0a2j7bxiO6DhZxTi4uylNi5OA
    - https://share.icloud.com/photos/03dx_QwFUV1lMI0m4St29ILPg
    - https://share.icloud.com/photos/03cWsB_jlq5fbsm-g0JikoXpQ
    - https://share.icloud.com/photos/047E4Zi0TXScme0ezAOXlviSQ
    - https://share.icloud.com/photos/040YRqq6OYDmwg-uKuDoQC_2A
    - <a href="https://www.flickr.com/photos/dalangalma/8199506986" title="Devil's Cataract video">Devil's Cataract video</a>” by <a href="https://www.flickr.com/photos/dalangalma/">Benjamin Hollis</a>, <a href="https://creativecommons.org/licenses/by/2.0/deed.en" rel="license noopener noreferrer">CC BY 2.0</a>)
    - “<a href="https://www.flickr.com/photos/ekilby/3954977794" title="Zakim Bridge Traffic Wide (HD)">Zakim Bridge Traffic Wide (HD)</a>” by <a href="https://www.flickr.com/photos/ekilby/">Eric Kilby</a>, <a href="https://creativecommons.org/licenses/by-sa/2.0/deed.en" rel="license noopener noreferrer">CC BY-SA 2.0</a>
    - “<a href="https://www.flickr.com/photos/r_topor/52166638758" title="Milky Way Timelapse over ATCA">Milky Way Timelapse over ATCA</a>” by <a href="https://www.flickr.com/photos/r_topor/">Rodney Topor</a>, <a href="https://creativecommons.org/licenses/by-nc-sa/2.0/deed.en" rel="license noopener noreferrer">CC BY-NC-SA 2.0</a>
    - “<a href="https://www.flickr.com/photos/mualphachi/3809913910" title="Ashford A30 Traffic">Ashford A30 Traffic</a>” by <a href="https://www.flickr.com/photos/mualphachi/">Maxwell Hamilton</a>, <a href="https://creativecommons.org/licenses/by/2.0/deed.en" rel="license noopener noreferrer">CC BY 2.0</a>
3. Load selected video from internet source and proceed with pipeline.

Acceptance criteria:
- Repo can be public without large media files.
- App can run locally with user-provided sample video.