from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    ethnicity = SelectField('Ethnicity', choices=[
        (0, 'Underrepresented Minority'),
        (1, 'Not Underrepresented Minority')
    ], validators=[DataRequired()])
    gender = SelectField('Gender', choices=[
        (0, 'Male'),
        (1, 'Female'),
        (2, 'Other/Non-binary')
    ], validators=[DataRequired()])
    income_bracket = SelectField('Income Bracket', choices=[
        (0, '$0-30k'),
        (1, '$30k-60k'),
        (2, '$60k-100k'),
        (3, '$100k-200k'),
        (4, '$200k+')
    ], validators=[DataRequired()])
    school = StringField('College Name', validators=[DataRequired()])
    major = StringField('Intended Major', validators=[DataRequired()])
    app_round = SelectField('Application Round', choices=[
        (0, 'Early Decision/Action'),
        (1, 'Regular Decision')
    ], validators=[DataRequired()])
    gpa = SelectField('GPA (Unweighted 4.0 scale)', choices=[
        (0, 'Below 2.5'),
        (1, '2.5-2.99'),
        (2, '3.0-3.49'),
        (3, '3.5-3.79'),
        (4, '3.8-4.0')
    ], validators=[DataRequired()])
    ap_ib_courses = StringField('Number of AP/IB Courses', validators=[DataRequired()])
    ap_ib_scores = SelectField('Average AP/IB Scores', choices=[
        (0, 'No scores'),
        (1, 'Below 3(AP)/4(IB)'),
        (2, '3-3.9(AP)/4-4.9(IB)'),
        (3, '4-4.9(AP)/5-5.9(IB)'),
        (4, '5(AP)/6-7(IB)')
    ], validators=[DataRequired()])
    test_score = SelectField('Test Score (SAT/ACT)', choices=[
        (0, 'No score/below 1000 SAT/20 ACT'),
        (1, '1000-1190 SAT/20-23 ACT'),
        (2, '1200-1390 SAT/24-29 ACT'),
        (3, '1400-1490 SAT/30-33 ACT'),
        (4, '1500+ SAT/34+ ACT')
    ], validators=[DataRequired()])
    location = SelectField('Location', choices=[
        (0, 'Rural (<10k population)'),
        (1, 'Suburban (10k-100k)'),
        (2, 'Urban (>100k)')
    ], validators=[DataRequired()])
    state_status = SelectField('State Status', choices=[
        (0, 'In-State'),
        (1, 'Out-of-State')
    ], validators=[DataRequired()])
    legacy = SelectField('Legacy Status', choices=[
        (0, 'Legacy'),
        (1, 'Non-Legacy')
    ], validators=[DataRequired()])
    first_gen = SelectField('First Generation College Student', choices=[
        (0, 'Not First-Gen'),
        (1, 'First-Gen')
    ], validators=[DataRequired()])
    languages = StringField('Number of Languages Known', validators=[DataRequired()])
    hooks = TextAreaField('Hooks (if any)')
    extracurriculars = TextAreaField('The more detailed and comprehensive you are, the more accurate our result is!\nList all extracurricular involvements, including leadership roles, time commitments, major achievements, etc. If you do not do this with detail, you will receive an unusually high acceptance probability due to our model not being able to extract enough information.', validators=[DataRequired()])
    awards = TextAreaField('The more detailed and comprehensive you are, the more accurate our result is!\nList all awards and honors you will submit on your application. If you do not do this with detail, you will receive an unusually high acceptance probability due to our model not being able to extract enough information.', validators=[DataRequired()])
    submit = SubmitField('Predict')