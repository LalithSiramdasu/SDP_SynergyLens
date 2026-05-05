# Implementation

## UI Migration Update

The current UI includes compatible ideas from the reference Drug-Synergy-Prediction project while preserving SDP_SynergyLens backend contracts. The migrated frontend improvements are limited to templates, static CSS, and static JavaScript:

- richer prediction result display with confidence-style signal language, report download, local history, and next actions
- SHAP explanation UI with positive/negative contributor panels, totals, summary text, chart, feature table, and safety note
- staged batch upload with sample CSV download, preview stats, horizontal table scrolling, and current `/api/batch` CSV response handling
- dynamic prediction chat with suggestion chips, wrapped/animated message bubbles, Gemini multi-key failover metadata, and fallback answers after configured keys fail
- static browser-only Project Assistant for route, CSV, score-band, SHAP, and safety questions
- side-by-side drug metadata cards using the existing PubChem-backed route
- backend readiness badge powered by `/api/health`
- About/Transparency view populated by `/api/system-summary` and `/api/model-performance-summary`
- verification coverage for `/api/health`, `/api/cell-lines`, `/api/system-summary`, `/api/model-performance-summary`, and all existing core endpoints

No model files, dataset files, feature construction, or prediction scoring thresholds were replaced as part of this UI migration.

## 1. Overview of the Implemented System

The developed system, named **SynergyLens**, is a full-stack web application designed to predict the synergy score of a pair of drugs in a selected biological context. The application combines a trained **XGBoost-based machine learning model**, a **Flask backend**, an interactive **HTML/CSS/JavaScript frontend**, and multiple supporting datasets stored in the `models/` directory. In addition to prediction, the system also provides **model explainability using SHAP-style feature contribution analysis**, **drug metadata lookup**, **batch prediction through CSV upload**, and a **chat-based explanation module** powered by Gemini when an API key is available.

From an implementation perspective, the system is organized into four major modules:

1. **Backend application module**
2. **Machine learning and feature engineering module**
3. **Frontend user interface module**
4. **Data and model artifact module**

Each module is explained below in a thesis-style, implementation-oriented manner.

## 2. Backend Application Module

The backend of the system is implemented in `app.py`. This file serves as the central controller of the application and is responsible for system initialization, loading model artifacts, processing user requests, generating predictions, and returning results to the frontend.

### 2.1 Flask Application Setup

The system uses the Flask framework to implement the web server. During startup, the Flask application object is created, upload limits are configured, the project root is identified, and environment variables are loaded from the `.env` file. A secret key is also initialized for session-based storage, which is required by the conversational explanation module.

This initialization layer performs the following tasks:

- creates the Flask application instance
- sets the maximum upload size for CSV files
- loads environment variables such as `FLASK_SECRET_KEY` and `GEMINI_API_KEY`
- enables session storage for preserving chat context across requests

This setup ensures that the application can securely manage both standard prediction requests and multi-turn conversational interactions.

### 2.2 Artifact Loading at Startup

At application startup, all model-related files are loaded once into memory. This design reduces repeated disk access and improves runtime performance. The following artifacts are loaded:

- the trained XGBoost model
- the list of expected feature columns
- the drug fingerprint lookup table
- the cell line feature lookup table
- the drug ID-to-name mapping table
- the drug explanation metadata file
- the feature glossary file
- the optional cancer label encoder

By loading these objects globally during startup, the system avoids unnecessary re-initialization during every request and can respond faster to prediction and explanation queries.

### 2.3 API Layer

The backend exposes multiple REST-style endpoints. These APIs allow the frontend to communicate with the machine learning pipeline in a structured way.

#### a. Home Route

The `/` route renders the main interface stored in `templates/index.html`. Before rendering, it supplies:

- the list of available drugs
- the list of supported cell lines
- the list of known cancer types

Thus, the initial page is dynamically populated from real project data rather than hard-coded values.

#### b. Drug Search API

The `/api/drugs` endpoint supports autocomplete search for drug names. When the user types a query, the backend filters the drug mapping table and returns the top matching results. This improves usability by preventing manual entry of numeric drug IDs.

#### c. Prediction API

The `/api/predict` endpoint is responsible for single-sample prediction. It accepts:

- `drug1_id`
- `drug2_id`
- `cell_line`
- optional `cancer_type`

The endpoint builds the input feature vector, runs the trained model, interprets the numerical score into a clinical interaction category, and returns a structured JSON response containing:

- predicted synergy score
- interpretation label
- display color and level
- drug names
- selected biological context

#### d. Explainability API

The `/api/explain` endpoint computes the most influential features for the current prediction. It uses XGBoost contribution values or a generic SHAP fallback if required. The endpoint returns:

- top contributing features
- SHAP contribution values
- optional feature descriptions
- model base value
- predicted score

This module transforms the prediction process from a black-box output into an interpretable decision support component.

#### e. Chat API

The `/api/chat` endpoint provides a question-answer interaction layer. It combines:

- the predicted score
- drug metadata
- major explanatory features
- recent chat history stored in session

If Gemini is configured, the endpoint generates a natural-language biomedical answer. If Gemini is unavailable, the system falls back to rule-based template responses. This makes the module robust even in environments where external LLM access is not enabled.

#### f. Drug Information API

The `/api/drug_info/<drug_id>` endpoint retrieves additional compound information. It combines local project metadata with real-time chemical properties from the PubChem REST API. The response may include:

- drug name
- molecular formula
- molecular weight
- IUPAC name
- structure image URL

This module enriches the prediction environment with supporting pharmacological context.

#### g. Batch Prediction API

The `/api/batch` endpoint allows the user to upload a CSV file containing multiple drug-pair records. Each row is processed independently, and the backend appends:

- `synergy_score`
- `interpretation`
- `error` if applicable

The processed output is returned as a downloadable CSV file. This module is useful when the user needs to screen many combinations in one execution.

## 3. Machine Learning and Feature Engineering Module

The machine learning implementation is also located in `app.py`, primarily through helper functions that prepare input data and interpret model output.

### 3.1 Feature Vector Construction

The core feature engineering logic is implemented in the `build_feature_vector()` function. This function creates one model-ready input record from the selected:

- first drug
- second drug
- cell line
- optional cancer type

The function first locates:

- the molecular fingerprint of each selected drug from `drug_fingerprints_lookup.csv`
- the biological profile of the chosen cell line from `cell_line_features_lookup.csv`

It then merges these elements into a single row aligned with `FEATURE_COLS`, which represents the exact feature order used during model training.

This implementation is important because the trained model expects a fixed feature structure. Any missing values are filled with zero, and all fields are coerced to numeric form to avoid runtime type issues.

### 3.2 Numeric Parsing and Data Cleaning

The helper `_parse_numeric()` is responsible for converting different raw value formats into valid numeric values. Some stored feature values may appear as:

- simple numbers
- strings
- stringified arrays
- bracketed scientific notation

The helper normalizes these inputs into float values. This makes the pipeline robust against inconsistencies in saved lookup files and protects the prediction module from input formatting errors.

### 3.3 Cancer Type Encoding

If the optional cancer encoder file is available, the chosen cancer type is transformed into a numerical representation through `cancer_encoder.transform()`. This preserves compatibility with model versions trained using encoded cancer labels. If the encoder is not available, the application still works using the remaining biological and molecular features.

### 3.4 Prediction and Score Interpretation

Once the feature vector is generated, the backend uses the loaded XGBoost model to compute the synergy score. The raw numerical result is then processed through `interpret_score()`, which maps the output into meaningful qualitative categories:

- Strong Synergy
- Moderate Synergy
- Mild Synergy
- Neutral / Additive
- Antagonistic

This categorical interpretation improves readability for end users who may not want to rely only on the raw numerical score.

## 4. Explainability Module

The explainability module is implemented using XGBoost contribution values and SHAP-compatible reasoning logic. Its purpose is to justify why the model produced a particular synergy score.

### 4.1 Booster Configuration Patch

Immediately after the model is loaded, the implementation checks the XGBoost booster configuration and corrects a possible `base_score` formatting issue. This patch is necessary because some saved boosters may store the base score in bracketed string form. The corrected configuration is reloaded into the booster before explanation is computed.

This is a practical implementation safeguard that improves compatibility between saved model artifacts and the runtime environment.

### 4.2 Contribution Extraction

The function `get_feature_contributions()` computes feature-level contribution values. When the model supports tree-based contributions, the system uses:

- `booster.predict(..., pred_contribs=True)`

This yields contribution scores for each input feature along with a base value. The top features are then selected by absolute magnitude and converted into a structured list containing:

- human-readable feature name
- raw feature name
- glossary description
- SHAP value

### 4.3 Human-Readable Feature Labels

Raw model feature names are not directly suitable for end-user interpretation. Therefore, the function `label_feature()` transforms technical column names into readable labels. For example:

- `drug1` becomes `Drug 1 identity`
- `drug2` becomes `Drug 2 identity`
- `cancer_encoded` becomes `Cancer type`
- fingerprint and RNAi columns are reformatted into more understandable text

This design makes the explanation output more appropriate for both visual charts and thesis documentation.

### 4.4 Rule-Based Explanation Text

The function `shap_to_text()` converts technical contribution values into plain-language explanation statements. Instead of exposing raw machine learning terminology alone, it produces sentences such as:

- molecular structure features increased the interaction
- biological pathway activity decreased the effect

This module acts as a bridge between numerical interpretability and user-friendly explanation.

## 5. Conversational Explanation Module

The application includes an interactive biomedical explanation layer that is implemented through several helper functions and the `/api/chat` endpoint.

### 5.1 Intent Detection

The function `classify_question()` analyzes the user question and categorizes it into intent groups such as:

- explanation
- score interpretation
- feature importance
- interaction reasoning
- general query

This simple intent classification helps the system decide how to shape a meaningful response.

### 5.2 Context Construction

The functions `build_cached_chat_context()`, `build_chat_context_text()`, and `build_conversation_prompt()` generate the textual context needed for question answering. The context includes:

- current prediction score
- interpretation category
- selected cell line and cancer type
- drug mechanism and class information
- top explanatory features
- recent dialogue history

The context is cached in the Flask session so that repeated questions about the same drug pair do not require the system to recompute the entire prediction pipeline each time.

### 5.3 LLM and Fallback Design

The implementation supports two response generation paths:

1. **Gemini-backed generation**, if the API key and SDK are available
2. **Fallback rule-based response generation**, if Gemini is not configured

This dual-path design improves reliability. Even when the external language model is unavailable, the chat feature still produces informative answers using the local explanation logic.

## 6. Frontend User Interface Module

The frontend is divided into three implementation layers:

- HTML structure in `templates/index.html`
- visual styling in `static/css/app.css`
- interactive behavior in `static/js/app.js`

### 6.1 HTML Interface Structure

The HTML file defines a multi-section workspace-based interface. The user interface is organized into the following major tabs:

- Predict
- Explain
- Chat
- Drug Info
- Batch

Each tab corresponds to a major functional module of the system. The page also includes:

- a hero section for application identity
- theme toggle controls
- metric summary cards
- forms for user input
- containers for charts, results, and downloadable batch output

This structure makes the system feel like a unified analytical workspace rather than a collection of separate screens.

### 6.2 JavaScript Interaction Layer

The frontend logic in `static/js/app.js` manages all dynamic behavior. Its responsibilities include:

- page navigation between tabs
- light/dark theme switching with local storage persistence
- drug autocomplete requests
- prediction API calls
- SHAP explanation API calls
- chat message handling
- drug information retrieval
- CSV upload and batch processing
- result rendering and alert handling

Functions such as `runPredict()`, `runExplain()`, and `runChat()` collect form inputs, call backend endpoints using the Fetch API, and update the interface without reloading the page. Therefore, the application behaves as a single-page analytical dashboard.

### 6.3 Visualization Logic

The frontend uses:

- an SVG gauge to display synergy score intensity
- Chart.js to plot SHAP feature contribution values

The function `renderPrediction()` updates the gauge, context labels, and interpretation summary. The function `renderExplanation()` constructs a horizontal bar chart where:

- positive contributions are shown in one color
- negative contributions are shown in another color

This design helps users quickly understand both the magnitude and direction of feature influence.

### 6.4 Batch Upload Interface

The JavaScript module also implements drag-and-drop CSV upload through `bindBatchUpload()` and `processBatchFile()`. This improves usability for high-volume prediction tasks and supports a more practical workflow for researchers.

## 7. Styling and Theme Module

The styling layer in `static/css/app.css` is implemented using modern CSS variables and modular section styling.

The stylesheet defines:

- color tokens for light and dark themes
- panel-based glassmorphism-inspired surfaces
- workspace grids and cards
- forms, buttons, and alerts
- chart and result containers
- animation effects and reveal transitions

The use of CSS custom properties allows the application to switch between light and dark themes without duplicating the entire stylesheet. The theme state is controlled from JavaScript and stored locally in the browser.

## 8. Data and Model Artifact Module

The `models/` directory stores the files required for prediction and interpretation. These artifacts are externalized from the code so that the analytical logic and trained resources remain loosely coupled.

The key files are:

- `drug_synergy_xgb_model.pkl`: trained XGBoost prediction model
- `feature_columns.pkl`: ordered feature list used during training
- `drug_fingerprints_lookup.csv`: structural descriptors for each drug
- `cell_line_features_lookup.csv`: biological features for each cell line
- `drug_name_id_map.csv`: mapping of drug IDs to human-readable names
- `explain_drug - Sheet1 (1).csv`: drug mechanism and class information
- `Feature_info - Data Dictionary.csv`: feature descriptions used in explanations

This modular storage approach makes it possible to update datasets or replace the model without rewriting the full application logic.

## 9. End-to-End Execution Flow

The implemented system follows the execution flow below:

1. The user opens the web interface.
2. Flask renders the page and injects available drugs, cell lines, and cancer types.
3. The user selects two drugs and a biological context.
4. The frontend sends the request to the backend.
5. The backend constructs a feature vector using lookup tables.
6. The trained XGBoost model predicts the synergy score.
7. The score is returned to the frontend with an interpretation label.
8. If requested, the backend computes feature contributions and sends them for chart rendering.
9. If the user opens the chat module, the system combines prediction context, drug information, and feature explanations to generate a biomedical response.
10. If batch mode is used, the same logic is applied iteratively to each CSV row and returned as a downloadable output file.

## 10. Summary of the Implementation

The implementation of SynergyLens demonstrates the integration of machine learning, explainable AI, biomedical metadata retrieval, and interactive web technologies into a single decision-support platform. The backend handles model loading, feature preparation, prediction, explanation, and batch execution. The frontend presents these capabilities through a modern workspace interface with dynamic charts, autocomplete, conversational assistance, and theme adaptability.

In summary, the project is not implemented as an isolated prediction script, but as a complete deployable analytical application in which each module contributes to usability, interpretability, and practical research support.
