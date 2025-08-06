# Visual Engagement Dynamics in B2B Cloud Computing Video Marketing

## Abstract
This repository provides the research framework and code for analyzing the relationship between visual elements, human presence, and audience engagement in B2B cloud computing video marketing. Using a dataset of 25,000 videos from AWS, Google Cloud, and IBM Cloud (2014-2024), we employ machine learning techniques for feature extraction and regression analysis to test a U-shaped engagement hypothesis.

## Research Questions
1. How do visual elements and human presence influence B2B video engagement metrics?
2. Does the relationship between visual elements and engagement follow a U-shaped curve?
3. How do narrative elements moderate this relationship?

## Methodology
- **Sample:** 25,000 videos from AWS, Google Cloud, and IBM Cloud (2014-2024).
- **Data Collection:** YouTube API, Python web scraping, Google Cloud Vision API.
- **Feature Extraction:**
  - Computer Vision: Scene analysis, facial recognition.
  - Audio: Fundamental frequency extraction.
  - NLP: Sentiment analysis of transcripts.
  - Optical Flow: Motion analysis.
- **Analysis:**
  - Exploratory Data Analysis.
  - Machine Learning Pipeline: Regression models to test hypotheses.

## Repository Structure
```
B2B-engagement/
├── analysis/
│   ├── audio_functions.py
│   ├── core_functions.py
│   ├── dataVideosYoutube.py
│   ├── image_functions.py
│   ├── text_functions.py
├── input/
│   ├── B2BcloudChannels.csv
│   ├── downloadVideo.csv
├── output/
│   ├── analysisYoutube.csv
│   ├── B2BcloudVideos.csv
│   ├── genderAnalysis.csv
│   ├── audio/
│   │   ├── *.wav
│   ├── json/
│   │   ├── *_evaluation.json
│   ├── text/
│   │   ├── *.txt
│   ├── video/
│   │   ├── *.mp4
├── test/
│   ├── env_check.py
│   ├── youtube_helper.py
├── dataAnalysis.ipynb
├── videoAnalysis.py
├── requirements.txt
├── README.md
```

## Key Features
- **Audio Analysis:** Extracts fundamental frequency and emotional valence from audio.
- **Video Analysis:** Scene detection, motion analysis, and human presence detection.
- **Text Analysis:** Sentiment analysis and keyword extraction from transcripts.
- **Data Integration:** Combines visual, audio, and textual features for comprehensive analysis.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/juanguerralatam/B2B-engagement.git
   ```
2. Navigate to the project directory:
   ```bash
   cd B2B-engagement
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run `dataAnalysis.ipynb` for exploratory data analysis.
- Use `videoAnalysis.py` for end-to-end video analysis.
- Test the environment setup with `test/env_check.py`.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
