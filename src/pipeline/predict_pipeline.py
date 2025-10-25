import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_model_object


class PredictPipeline:
    def __init(self):
        pass

    def predict(self,data_frame):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_model_object(file_path=model_path)
            preprocessor = load_model_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(data_frame)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
             Hours_Studied: float,
             Attendance: float,
             Sleep_Hours: float,
             Previous_Scores: float,
             Tutoring_Sessions: float,
             Physical_Activity: float,
             Parental_Involvement: str,
             Access_to_Resources: str,
             Extracurricular_Activities: str,
             Motivation_Level: str,
             Internet_Access: str,
             Family_Income: str,
             School_Type: str,
             Peer_Influence: str,
             Learning_Disabilities: str,
             Gender: str):
    
            # Numerical features
            self.Hours_Studied = Hours_Studied
            self.Attendance = Attendance
            self.Sleep_Hours = Sleep_Hours
            self.Previous_Scores = Previous_Scores
            self.Tutoring_Sessions = Tutoring_Sessions
            self.Physical_Activity = Physical_Activity

            # Categorical features
            self.Parental_Involvement = Parental_Involvement
            self.Access_to_Resources = Access_to_Resources
            self.Extracurricular_Activities = Extracurricular_Activities
            self.Motivation_Level = Motivation_Level
            self.Internet_Access = Internet_Access
            self.Family_Income = Family_Income
            self.School_Type = School_Type
            self.Peer_Influence = Peer_Influence
            self.Learning_Disabilities = Learning_Disabilities
            self.Gender = Gender

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "Hours_Studied": [self.Hours_Studied],
            "Attendance": [self.Attendance],
            "Sleep_Hours": [self.Sleep_Hours],
            "Previous_Scores": [self.Previous_Scores],
            "Tutoring_Sessions": [self.Tutoring_Sessions],
            "Physical_Activity": [self.Physical_Activity],
            "Parental_Involvement": [self.Parental_Involvement],
            "Access_to_Resources": [self.Access_to_Resources],
            "Extracurricular_Activities": [self.Extracurricular_Activities],
            "Motivation_Level": [self.Motivation_Level],
            "Internet_Access": [self.Internet_Access],
            "Family_Income": [self.Family_Income],
            "School_Type": [self.School_Type],
            "Peer_Influence": [self.Peer_Influence],
            "Learning_Disabilities": [self.Learning_Disabilities],
            "Gender": [self.Gender]
}

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)