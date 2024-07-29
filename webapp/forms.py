from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    school = StringField('College Name', validators=[DataRequired()])
    app_round = StringField('EA/RD', validators=[DataRequired()])
    major = StringField('Intended Major/s', validators=[DataRequired()])
    info = TextAreaField('The more detailed and comprehensive you are, the more accurate our result is! Note that if you are not detailed enough in ECs or Awards, you will get an abnormaly high probability of acceptance due to the lack of ability to extract enough data for the model.', 
                         validators=[DataRequired()],
                         default='''Demographics

- Gender:

- Race/Ethnicity:

- Residence:

- Income Bracket:

- Type of School:

- Hooks (Recruited Athlete, URM, First-Gen, Geographic, Legacy, etc.):

Academics

- GPA (UW/W):

- Rank (or percentile):

- # of Honors/AP/IB/Dual Enrollment/etc.:

- Senior Year Course Load:

Standardized Testing

List the highest scores earned and all scores that were reported.

- SAT I: #### (###RW, ###M)

- ACT: ## (##E, ##M, ##R, ##S)

- AP/IB: ____ (#), ____ (#), ...

Other (ex. IELTS, TOEFL, etc.):

Extracurriculars/Activities

List all extracurricular involvements, including leadership roles, time commitments, major achievements, etc.

1. #1

2. #2

3. #3

4. #4

5. #5

6. #6

7. #7

8. #8

9. #9

10. #10

Awards/Honors

List all awards and honors submitted on your application.

1. #1

2. #2

3. #3

4. #4

5. #5

Additional Information:

(anything of relevance)''')
    submit = SubmitField('Predict')