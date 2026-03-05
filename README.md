# Support Insights Dashboard

A local, reusable Streamlit dashboard for
- Generating synthetic support cases (Salesforce-style records)
- Scrubbing PII via token replacement
- Running sentiment + friction analytics
- Exporting a cleaned CSV for downstream workflows

## Why this exists
Support teams often track outcomes (CSATNPS) but don’t systematically analyze friction signals inside case text.  
This tool demonstrates a lightweight pipeline that turns case text into structured insights.

## Features
- Synthetic case generator (default 500 cases)
- ZoomInfo-like Tier-1 status workflow simulation
  - New → Awaiting Response → Responded → Reopened → Working → In Progress → Resolved
  - Open vs Closed logic baked in
- PII scrubber
  - Replaces emails, phones, and names with tokens
  - Shows PII removal metrics
- Sentiment analysis (Subject + Description)
- Analytics dashboards
  - Status + OpenClosed distribution
  - Category distribution
  - Escalations by category
  - Negative sentiment by category
  - Repeat-account signals
- Export cleaned dataset to CSV

## Run locally (Windows PowerShell)

```powershell
cd $envUSERPROFILEDocumentssupport_insights_dashboard
..venvScriptsActivate.ps1
pip install -r requirements.txt
streamlit run app.py