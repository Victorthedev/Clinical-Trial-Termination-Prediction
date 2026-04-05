import requests
import pandas as pd
import time
import os

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_trials(status, max_records=5000):
    trials = []
    next_page_token = None
    
    print(f"Fetching {status} trials...")
    
    while len(trials) < max_records:
        params = {
            "filter.overallStatus": status,
            "pageSize": 100,
            "fields": "NCTId,OverallStatus,Phase,EnrollmentCount,LeadSponsorClass,StartDate,CompletionDate,StudyType,Condition"
        }
        
        if next_page_token:
            params["pageToken"] = next_page_token
            
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
            
        data = response.json()
        studies = data.get("studies", [])
        
        if not studies:
            break
            
        for study in studies:
            proto = study.get("protocolSection", {})
            id_module = proto.get("identificationModule", {})
            status_module = proto.get("statusModule", {})
            design_module = proto.get("designModule", {})
            sponsor_module = proto.get("sponsorCollaboratorsModule", {})
            conditions_module = proto.get("conditionsModule", {})
            
            trials.append({
                "nct_id": id_module.get("nctId"),
                "status": status_module.get("overallStatus"),
                "phase": design_module.get("phases", [None])[0],
                "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                "sponsor_class": sponsor_module.get("leadSponsor", {}).get("class"),
                "start_date": status_module.get("startDateStruct", {}).get("date"),
                "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                "study_type": design_module.get("studyType"),
                "condition": conditions_module.get("conditions", [None])[0]
            })
            
        print(f"  Fetched {len(trials)} so far...")
        next_page_token = data.get("nextPageToken")
        
        if not next_page_token:
            break
            
        time.sleep(0.5)
    
    return trials[:max_records]

def main():
    os.makedirs("data/raw", exist_ok=True)
    
    completed = fetch_trials("COMPLETED", max_records=5000)
    terminated = fetch_trials("TERMINATED", max_records=5000)
    
    df = pd.DataFrame(completed + terminated)
    
    output_path = "data/raw/trials_raw.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nDone. Total records: {len(df)}")
    print(f"Saved to {output_path}")
    print(df["status"].value_counts())

if __name__ == "__main__":
    main()