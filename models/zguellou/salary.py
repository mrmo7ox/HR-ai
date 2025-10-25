import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('salary_data.csv')

df.drop('id', axis=1, inplace=True)

df = df.dropna(subset=['years_experience', 'salary_mad'])

df['role'] = df['role'].str.title()
df['degree'] = df['degree'].str.title()
df['company_size'] = df['company_size'].str.title()
df['location'] = df['location'].str.title()
df['level'] = df['level'].str.title()

# rtebt had lfields bach y3tina encoding mgad
level_order = ['Intern', 'Junior', 'Mid', 'Senior', 'Lead']
company_size_order = ['Small', 'Mid', 'Large', 'Enterprise']
degree_order = ['No Degree', 'Bachelors', 'Masters', 'Phd']

df['level_encoded'] = df['level'].map({level: i for i, level in enumerate(level_order)})
df['company_size_encoded'] = df['company_size'].map({size: i for i, size in enumerate(company_size_order)})
df['degree_encoded'] = df['degree'].map({deg: i for i, deg in enumerate(degree_order)})

label_encoders = {}
nominal_cols = ['role', 'location']

for col in nominal_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

label_encoders['level'] = {level: i for i, level in enumerate(level_order)}
label_encoders['company_size'] = {size: i for i, size in enumerate(company_size_order)}
label_encoders['degree'] = {deg: i for i, deg in enumerate(degree_order)}

categorical_cols = ['role', 'degree', 'company_size', 'location', 'level']
feature_cols = ['years_experience'] + [col + '_encoded' for col in categorical_cols]
X = df[feature_cols]
y = df['salary_mad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))


# ========== PREDICTION FUNCTION ==========
def predict_salary(years_exp, role, degree, company_size, location, level):
    """Predict salary with proper encoding"""
    input_data = {
        'years_experience': years_exp,
        'role_encoded': label_encoders['role'].transform([role])[0],
        'degree_encoded': label_encoders['degree'][degree],
        'company_size_encoded': label_encoders['company_size'][company_size],
        'location_encoded': label_encoders['location'].transform([location])[0],
        'level_encoded': label_encoders['level'][level]
    }
    input_df = pd.DataFrame([input_data])[feature_cols]
    return model.predict(input_df)[0]

salary = predict_salary(4, 'Ml Engineer', 'Masters', 'Mid', 'Marrakech', 'Mid')
salary1 = predict_salary(4, 'Ml Engineer', 'Masters', 'Mid', 'Agadir', 'Mid')
print(f"Predicted Salary: {salary:,.2f} MAD")
print(f"Predicted Salary: {salary1:,.2f} MAD")