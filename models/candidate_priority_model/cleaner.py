import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from trainer import training_model

PATH = 'candidate_priority.csv'

def strip_and_capitalize(value):
    value = str(value).strip()
    value = value.lower().capitalize()
    return value

def parse_years_exp(col):
    regex = r'^(\d+)[\-–](\d+)$'
    def convert(value):
        value = str(value).strip()
        if value.endswith('+'):
            return int(value.replace('+', ''))
        m = re.match(regex, value)
        if m:
            v1 = int(m.group(1))
            v2 = int(m.group(2))
            return (v1 + v2) / 2
    return col.apply(convert)


def parse_skills_coverage(col):
    mapp = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }
    def convert(value):
        value = strip_and_capitalize(value)
        for key, v in mapp.items():
            if key == value:
                return v
    return col.apply(convert)
        

def parse_referral_flag(col):
    def convert(value):
        return int(value)
    return col.apply(convert)


def parse_english_level(col):
    le = LabelEncoder()
    s = []
    for value in col:
        value = strip_and_capitalize(value)
        s.append(value)
    encoded = le.fit_transform(s)
    return encoded


def parse_location_match(col):
    le = LabelEncoder()
    s = []
    for value in col:
        value = strip_and_capitalize(value)
        if value == 'Remoteok':
            value = 'RemoteOK'
        s.append(value)
    encoded = le.fit_transform(s)
    return encoded


def parse_priority(col):
    s = []
    for value in col:
        value = strip_and_capitalize(value)
        s.append(value)
    return s


def main():
    data = pd.read_csv(PATH)
    saved_data = data.copy()
    saved_data = saved_data.dropna()
    saved_data = saved_data.drop('id', axis=1)
    saved_data['years_exp_band'] = parse_years_exp(saved_data['years_exp_band'])
    saved_data['skills_coverage_band'] = parse_skills_coverage(saved_data['skills_coverage_band'])
    saved_data['referral_flag'] = parse_referral_flag(saved_data['referral_flag'])
    saved_data['english_level'] = parse_english_level(saved_data['english_level'])
    saved_data['location_match'] = parse_location_match(saved_data['location_match'])
    saved_data['priority'] = parse_priority(saved_data['priority'])
    
    training_model(saved_data=saved_data)


if __name__ == '__main__':
    main()