# Visual Engagement Dynamics in B2B Cloud Computing Video Marketing: A Machine Learning Analysis of the Non-Linear Relationship Between Visual Elements and Audience Engagement

**Research Repository for Journal of Business-to-Business Marketing**

## Abstract

This repository contains the complete research framework and analytical code for examining the non-linear relationship between visual elements, human presence, and audience engagement in B2B cloud computing video marketing. Using a comprehensive dataset of 25,000 videos from leading cloud service providers (Amazon Web Services, Google Cloud Platform, and IBM Cloud) spanning approximately 10 years (2014-2024), we employ machine learning techniques for feature extraction followed by regression analysis to test the hypothesis of a U-shaped phenomenon in visual engagement dynamics.

## Research Questions

**Primary Research Question:** How do visual elements, human presence, and storytelling components in B2B cloud computing videos influence audience engagement metrics, and does this relationship follow a non-linear pattern?

**Secondary Research Questions:**
1. What is the optimal combination of visual complexity and human presence for maximizing B2B video engagement?
2. Does the relationship between visual elements and engagement exhibit a U-shaped curve, suggesting diminishing returns followed by renewed effectiveness?
3. How do narrative elements moderate the relationship between visual features and audience engagement in B2B contexts?

## Theoretical Framework

This study builds upon the **Elaboration Likelihood Model (ELM)** and **Dual-Process Theory** in marketing communications, examining how visual complexity in B2B videos influences cognitive processing and subsequent engagement behaviors. We hypothesize that visual elements follow an inverted-U relationship with engagement, where moderate visual complexity optimizes attention and comprehension, while excessive complexity creates cognitive overload.

**Key Hypotheses:**
- **H1:** Visual complexity exhibits a curvilinear (U-shaped) relationship with B2B video engagement
- **H2:** Human presence moderates the relationship between visual elements and engagement
- **H3:** Narrative quality amplifies the positive effects of optimal visual-human presence combinations

## Methodology

### Research Design
This study employs a **quantitative experimental design** using archival data analysis of B2B video content. We utilize a **longitudinal cross-sectional approach** examining video performance across multiple time periods and platforms.

### Data Collection
- **Sample:** 25,000 B2B cloud computing videos
- **Sources:** Amazon Web Services (AWS), Google Cloud Platform, IBM Cloud
- **Time Period:** January 2014 - December 2024 (10-year span)
- **Sampling Method:** Complete enumeration of publicly available videos from official corporate channels
- **Data Collection Tools:** YouTube API v3, Python web scraping, Google Cloud Vision API

### Feature Extraction Methodology
- **Computer Vision:** Google Cloud Vision API for facial recognition, object detection, scene analysis
- **Audio Processing:** Python-based fundamental frequency (F0) extraction using Praat integration
- **Natural Language Processing:** Sentiment analysis using VADER and TextBlob for transcript analysis
- **Optical Flow Analysis:** OpenCV implementation for motion magnitude and direction calculation

### Statistical Analysis Plan
1. **Exploratory Data Analysis:** Descriptive statistics and correlation matrices
2. **Feature Engineering:** Principal Component Analysis for dimensionality reduction
3. **Machine Learning Pipeline:** 
   - Feature extraction using ensemble methods (Random Forest, Gradient Boosting)
   - Cross-validation with 80/20 train-test split
4. **Regression Analysis:** 
   - Linear and polynomial regression models
   - Robust standard errors clustered by video channel
   - Control for temporal effects and platform-specific characteristics

## Variable Operationalization

| Variable                | Operationalization                                                                 | Data Source           | Theoretical Foundation                          |
|-------------------------|-----------------------------------------------------------------------------------|-----------------------|-------------------------------------------------|
| **Dependent Variables** |                                                                                   |                       |                                                 |
| Engagement Score        | Composite metric: log(views + likes + comments), normalized by channel followers  | YouTube API v3        | Malik et al. (2024); Cheng & Zhang (2024)      |
| View Duration          | Average view duration as percentage of total video length                         | YouTube Analytics     | Li & Berger (2021)                              |
| **Primary Predictors**  |                                                                                   |                       |                                                 |
| Visual Complexity       | Weighted composite of saturation variance, scene transitions, and object density   | Computer Vision API   | Berlyne (1970); Pieters et al. (2010)          |
| Human Presence Index    | Binary indicator weighted by face detection confidence and screen time            | Google Cloud Vision   | Social Presence Theory (Short et al., 1976)    |
| Narrative Quality       | Sentiment consistency score and linguistic complexity measures                     | NLP Pipeline          | Transportation Theory (Green & Brock, 2000)    |
| **Control Variables**   |                                                                                   |                       |                                                 |
| Video Age               | Days since publication (log-transformed)                                          | YouTube API           | Temporal decay effects (Malik et al., 2024)    |
| Channel Authority       | Subscriber count (log-normalized) at time of video publication                    | YouTube API           | Source credibility (Hovland & Weiss, 1951)     |
| Video Length            | Duration in seconds (log-transformed)                                             | YouTube API           | Attention economics (Davenport & Beck, 2001)   |
| **Visual Features**     |                                                                                   |                       |                                                 |
| Scene Transitions       | Rate of scene changes per minute using shot boundary detection                     | OpenCV Analysis       | Cognitive Load Theory (Sweller, 1988)          |
| Color Saturation        | Average HSV saturation across all frames (0-1 scale)                             | Image Processing      | Color Psychology (Elliot & Maier, 2014)        |
| Brightness Variance     | Standard deviation of luminance values across video timeline                       | Image Processing      | Visual Attention Theory (Itti & Koch, 2001)    |
| Motion Dynamics         | Optical flow magnitude and directional consistency                                | Computer Vision       | Dynamic Visual Processing (Cutting et al., 2010)|
| **Human Elements**      |                                                                                   |                       |                                                 |
| Face Detection Count    | Average number of faces detected per scene (confidence > 0.8)                    | Google Cloud Vision   | Face Recognition Literature (Bruce & Young, 1986)|
| Gender Representation   | Binary classification of primary speaker gender                                   | Google Cloud Vision   | Gender Effects in Marketing (Meyers-Levy, 1989)|
| Facial Expression       | Smile detection probability score (0-1) averaged across appearances              | Google Cloud Vision   | Emotional Contagion Theory (Hatfield et al., 1994)|
| **Audio Features**      |                                                                                   |                       |                                                 |
| Vocal Pitch (F0)        | Fundamental frequency in Hz, extracted using Praat algorithms                     | Audio Processing      | Vocal Communication Research (Ohala, 1984)     |
| Speech Rate             | Words per minute calculated from transcript timing                                | Speech Recognition    | Processing Fluency (Alter & Oppenheimer, 2009) |
| **Content Analysis**    |                                                                                   |                       |                                                 |
| Sentiment Valence       | Weighted sentiment score using VADER lexicon (-1 to +1)                          | NLP Analysis          | Affective Response Theory (Russell, 1980)      |
| Technical Complexity    | Ratio of technical terminology to total word count                               | Text Mining           | Expertise Communication (Bromme et al., 2005)  |
| Call-to-Action Presence | Binary indicator of explicit action requests in video or description             | Text Mining           | Persuasion Theory (Cialdini, 2007)             |

## Expected Contributions

### Theoretical Contributions
1. **Extension of Visual Complexity Theory:** First empirical test of U-shaped visual engagement relationship in B2B contexts
2. **Integration of Dual-Process Theory:** Bridging cognitive psychology and B2B marketing through visual processing mechanisms
3. **Human Presence Framework:** Developing a comprehensive model for human elements in digital B2B communication

### Methodological Contributions
1. **Multi-Modal Analysis Pipeline:** Novel integration of computer vision, NLP, and audio processing for marketing research
2. **Large-Scale Video Analytics:** Advancing big data methodologies in B2B marketing research
3. **Temporal Analysis Framework:** Longitudinal approach to understanding video engagement evolution

### Practical Contributions
1. **B2B Video Strategy Guidelines:** Evidence-based recommendations for optimal visual complexity
2. **ROI Optimization Framework:** Quantitative models for predicting video engagement outcomes
3. **Industry Benchmarking:** Comparative analysis across major cloud service providers

## Repository Structure

```
├── data/
│   ├── raw/                    # Original video metadata and engagement metrics
│   ├── processed/              # Cleaned and feature-engineered datasets
│   └── external/               # Third-party data sources and APIs
├── src/
│   ├── data_collection/        # YouTube API and web scraping scripts
│   ├── feature_extraction/     # Computer vision and NLP processing
│   ├── analysis/               # Statistical models and ML pipelines
│   └── visualization/          # Plotting and reporting functions
├── models/
│   ├── trained_models/         # Serialized ML models
│   └── model_evaluation/       # Performance metrics and validation
├── notebooks/
│   ├── exploratory_analysis/   # EDA and data exploration
│   ├── modeling/               # Model development and testing
│   └── reporting/              # Results visualization and interpretation
├── docs/
│   ├── methodology/            # Detailed methodological documentation
│   ├── codebook/               # Variable definitions and coding schemes
│   └── api_documentation/      # Technical documentation
└── outputs/
    ├── figures/                # Publication-ready visualizations
    ├── tables/                 # Statistical results and summaries
    └── reports/                # Technical and executive summaries
```

## Installation and Usage

### Prerequisites
```bash
# Python environment setup
conda create -n b2b-video-analysis python=3.9
conda activate b2b-video-analysis

# Install required packages
pip install -r requirements.txt
```

### Data Collection
```bash
# Configure API credentials
cp config/config_template.yml config/config.yml
# Edit config.yml with your YouTube API key and Google Cloud credentials

# Run data collection pipeline
python src/data_collection/collect_youtube_data.py
python src/data_collection/extract_video_features.py
```

### Analysis Pipeline
```bash
# Feature engineering and preprocessing
python src/analysis/preprocess_data.py

# Run machine learning models
python src/analysis/train_models.py

# Generate results and visualizations
python src/analysis/generate_results.py
```

## Ethical Considerations

This research adheres to ethical guidelines for digital marketing research:
- **Public Data Only:** Analysis limited to publicly available YouTube videos
- **No Personal Identification:** Individual viewers are not identified or tracked
- **Platform Compliance:** All data collection respects YouTube's Terms of Service and API limitations
- **Privacy Protection:** No personally identifiable information is collected or stored
- **Fair Use:** Academic research purposes under fair use provisions

## Data Availability Statement

**Code Availability:** All analysis code is publicly available in this repository under MIT License.

**Data Availability:** The complete dataset will be made available upon publication through [specify repository - e.g., Harvard Dataverse, figshare] in accordance with YouTube's data sharing policies and academic research standards.

**Replication Package:** A complete replication package including processed data (where permissible), analysis scripts, and documentation will be provided to enable full reproducibility of results.

## Citation

### Repository Citation
```
Guerra, J. (2025). Visual Engagement Dynamics in B2B Cloud Computing Video Marketing: 
A Machine Learning Analysis [Computer software]. GitHub. 
https://github.com/juanguerralatam/B2B-engagement
```

### Dataset Citation
```
Guerra, J. (2025). B2B Cloud Computing Video Engagement Dataset, 2014-2024 
[Dataset]. [Repository to be specified upon publication].
```

### Paper Citation (When Published)
```
Guerra, J. (2025). Visual engagement dynamics in B2B cloud computing video marketing: 
A machine learning analysis of the non-linear relationship between visual elements 
and audience engagement. Journal of Business-to-Business Marketing, [Volume(Issue)], [pages].
```

## Contact Information

**Principal Investigator:** Juan Guerra  
**Institution:** [Your Institution]  
**Email:** [Your Email]  
**ORCID:** [Your ORCID ID]

## Funding Acknowledgments

[Include funding sources and grant numbers if applicable]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Academic use is encouraged with proper citation. Commercial use requires explicit permission.

## References

Alter, A. L., & Oppenheimer, D. M. (2009). Uniting the tribes of fluency to form a metacognitive nation. *Personality and Social Psychology Review*, 13(3), 219-235.

Berger, J., & Milkman, K. L. (2012). What makes online content viral? *Journal of Marketing Research*, 49(2), 192-205.

Berlyne, D. E. (1970). Novelty, complexity, and hedonic value. *Perception & Psychophysics*, 8(6), 279-286.

Bromme, R., Rambow, R., & Nückles, M. (2005). Expertise and estimating what other people know: The influence of professional experience and type of knowledge. *Journal of Experimental Psychology: Applied*, 7(4), 317-330.

Bruce, V., & Young, A. (1986). Understanding face recognition. *British Journal of Psychology*, 77(3), 305-327.

Cheng, X., & Zhang, Y. (2024). Visual complexity and engagement in social media marketing: Evidence from video content analysis. *Journal of Interactive Marketing*, 58(2), 45-62.

Cialdini, R. B. (2007). *Influence: The psychology of persuasion* (Rev. ed.). Harper Business.

Cutting, J. E., DeLong, J. E., & Nothelfer, C. E. (2010). Attention and the evolution of Hollywood film. *Psychological Science*, 21(3), 432-439.

Davenport, T. H., & Beck, J. C. (2001). *The attention economy: Understanding the new currency of business*. Harvard Business Review Press.

Elliot, A. J., & Maier, M. A. (2014). Color psychology: Effects of perceiving color on psychological functioning in humans. *Annual Review of Psychology*, 65, 95-120.

Fong, L., Chen, H., & Wang, S. (2025). Emotional arousal and B2B video marketing effectiveness: A multimodal analysis approach. *Industrial Marketing Management*, 89, 156-168.

Green, M. C., & Brock, T. C. (2000). The role of transportation in the persuasiveness of public narratives. *Journal of Personality and Social Psychology*, 79(5), 701-721.

Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1994). *Emotional contagion*. Cambridge University Press.

Hovland, C. I., & Weiss, W. (1951). The influence of source credibility on communication effectiveness. *Public Opinion Quarterly*, 15(4), 635-650.

Itti, L., & Koch, C. (2001). Computational modelling of visual attention. *Nature Reviews Neuroscience*, 2(3), 194-203.

Jalali, A., & Papatla, P. (2019). The effect of promotional content on B2B social media engagement: Evidence from LinkedIn. *Journal of Business Research*, 101, 165-178.

Li, J., & Berger, J. (2021). Attention, please: The effect of visual attention on video engagement. *Journal of Consumer Research*, 48(1), 87-104.

Li, X., & Xie, H. (2020). Human presence in B2B video marketing: Effects on trust and purchase intention. *International Journal of Business Communication*, 57(3), 412-431.

Luo, R., Kim, S., & Park, J. (2024). Color saturation and brightness effects on video engagement: A computer vision approach. *Marketing Science*, 43(2), 234-251.

Malik, A., Singh, P., & Johnson, M. (2024). Temporal dynamics of video engagement: A longitudinal analysis of B2B content. *Journal of Marketing Analytics*, 12(1), 78-95.

Meyers-Levy, J. (1989). Gender differences in information processing: A selectivity interpretation. In P. Cafferata & A. Tybout (Eds.), *Cognitive and affective responses to advertising* (pp. 219-260). Lexington Books.

Ohala, J. J. (1984). An ethological perspective on common cross-language utilization of F0 of voice. *Phonetica*, 41(1), 1-16.

Pieters, R., Wedel, M., & Batra, R. (2010). The stopping power of advertising: Measures and effects of visual complexity. *Journal of Marketing*, 74(5), 48-60.

Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

Short, J., Williams, E., & Christie, B. (1976). *The social psychology of telecommunications*. John Wiley & Sons.

Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257-285.

Walker, R., Johnson, L., & Davis, K. (2017). Social media engagement metrics in B2B contexts: A comprehensive analysis. *Journal of Business & Industrial Marketing*, 32(6), 847-861.

Yang, F., Thompson, A., & Rodriguez, C. (2025). Gender representation and audience engagement in corporate video marketing. *Psychology & Marketing*, 42(1), 112-128.
