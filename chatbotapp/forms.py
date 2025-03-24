from django import forms

class PredictForm(forms.Form):
    RANK_CHOICES = [(i, str(i)) for i in range(0, 100000, 1000)]
    
    rank = forms.IntegerField(
        label='Your Rank',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    gender = forms.ChoiceField(
        choices=[('M', 'Male'), ('F', 'Female')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    caste = forms.ChoiceField(
        choices=[('OC', 'OC'), ('BC_A', 'BC_A'), ('BC_B', 'BC_B'), ('SC', 'SC'), ('ST', 'ST')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    branch = forms.ChoiceField(
        choices=[
            ('CSE', 'Computer Science'),
            ('ECE', 'Electronics'),
            ('MEC', 'Mechanical'),
            ('CIV', 'Civil')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )




from django import forms

class InputForm(forms.Form):
    rank = forms.IntegerField(label='Rank')
    gender = forms.CharField(label='Gender')
    allocated_category = forms.CharField(label='Allocated Category')
    region = forms.CharField(label='Region')
