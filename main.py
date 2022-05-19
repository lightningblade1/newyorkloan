import missingno as missingno
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from sklearn.preprocessing import LabelEncoder

loan = pd.read_csv("sample.csv")
# Exploratory Data Analysis
loan = loan.replace({'applicant_race_name_1': {
    'Information not provided by applicant in mail, Internet, or telephone application': "Information unavailable", }})
missingno.matrix(loan)
loan = loan.dropna(subset=(
    ['applicant_income_000s',
     "population"]))  # removing population rows gets rid of multiple 4 NA rows in multiple columns
loan = loan.reset_index(drop=True)
plt.figure(figsize=(5, 3))
dataplot = sns.heatmap(loan.corr(), cmap="YlGnBu", annot=True)
plt.setp(dataplot.get_xticklabels(), rotation=10, horizontalalignment='right')
#number_of_owner_occupied_units show high correlation with number_of_1_to_4_family_units,population and minority_population
#loan amount and applicant income are also highly correlated
plt.show()

ax=sns.lmplot(data=loan, x="number_of_owner_occupied_units", y="population")
plt.show()
#number_of_owner_occupied_units and population show a linear relationship. Meaning where number of owner occupied units are high the population in those area is also dense.
sns.lmplot(data=loan, x="number_of_owner_occupied_units", y="number_of_1_to_4_family_units")
plt.show()
#Similar Relationship as above can be seen between number_of_owner_occupied_units and number_of_1_to_4_family_units.This could be because the owner of these houses live at their homes along with their families
#This can also be used to explain the dense population phenomenon.
sns.lmplot(data=loan, x="number_of_owner_occupied_units", y="minority_population")
plt.show()
#The above plot show that number_of_owner_occupied_units is negatively corelated with minority_population.
sns.lmplot(data=loan, x="loan_amount_000s", y="applicant_income_000s")
plt.show()
# It can be seen that applicant_income_000s and loan_amount_000s are positively correlated.However, it can be seen that the data regression plot has some standard deviation to it which we will trx explain now
sns.countplot(x="applicant_race_name_1", data=loan)
# The plot shows that the dataset is skewed where it has very dispropotionately data points for applicants of white ethnicity
sns.lmplot(data=loan[(loan.applicant_race_name_1 != "Information unavailable")], x="loan_amount_000s",
           y="applicant_income_000s", truncate=False, hue="applicant_race_name_1", scatter_kws={'alpha': 0.5}).set(
    title='Black')
plt.show()
# Looking at the slope it can be seen that ethnicity has a role in how much loan would people of certain ethnicity apply for with respect to their income.It can be seen that American indian have the lowest slope
# However,people with white ethnicity have the highest slope

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="applicant_race_name_1", y="applicant_income_000s", data=loan, capsize=0.2)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.show()
# The bar plot shows that ethnicity has effect on applicant income as well
fig = plt.figure(figsize=(20, 1))
sns.countplot(y='denial_reason_name_1', data=loan).set(
    title='Denial Reason 1');  # This plot shows that debt to income ratio is the biggest reason for loans being denied
fig = plt.figure(figsize=(20, 1))
sns.countplot(y='denial_reason_name_2', data=loan).set(
    title='Denial Reason 2');  # This pot shows that debt to income ratio is the 2nd biggest reason for loans being denied
plt.show()
loan["debt_to_income"] = loan["loan_amount_000s"] / (loan["applicant_income_000s"])  # debt to income ratio calculation
loan["debt_to_income"] = loan["debt_to_income"].round(3)
fig = plt.figure(figsize=(20, 1))
ax = sns.countplot(x="action_taken_name", data=loan,
                   capsize=0.2)  # Data points for loan originated is much higher than other values.

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="action_taken_name", y="debt_to_income", data=loan, capsize=0.2)
# when considering debt to income ratio it can be seen that most loans were originated between a debt to income ratio of 2.3 to 2.5 with a mean at 2.4.
# now we look at application approved not accepted. When considering debt to income ratio it needs to be kept in mind
# that the higher it gets. the more it will be difficult for the applicant to make monthly payments.So in this case the HMDA loan might be approved from the side of the financial institution.
# However, this doesnt necessarily mean the loan will be given to the applicant.For example if the property is in an area that is flooded or at high risk of floods
# The financial institution might need the applicant to buy flood insurance. If the applicant refuses to buy the flood insurance or simply refuses to show up to recieve the loan or sign documents. The loan will be given in the category approved but not accepted
# Approved but not accepted has a mean of 2.7 and a S.D between 2.6 and 2.8.The fact the S.D of "application approved not accepted" starts when "Loan originated" ends might be because as the debt to income ratio incereased above 2.5 the applicants were more likely not able to
# pay for insurance or other things that financial institution needed to issue the loan.
# Application withdrawn by applicant doesnt necessarily mean that they didnt get loan. They could have applied for the loan with a lower loan amount to increase the probability of their loan being approved.However the ways we can identify those applicants from this dataset are limited.Notice how application withdrawn and
# aplication denied bars are so similar so it is likely that the participants were advised to withdraw their application and apply with a lower amount.

for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)

tree = loan[(loan["action_taken_name"] == "Application denied by financial institution") & (
        loan["action_taken_name"] == "Loan originated")]
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
#plt.show()

plt.figure(figsize=(5, 3))
sns.lmplot(data=loan[
    ((loan["action_taken_name"] == "Loan originated") | (
                loan["action_taken_name"] == "Application denied by financial institution"))], x="loan_amount_000s",
           y="applicant_income_000s", hue="action_taken_name")
#As can be seen by the linear regression model that the slope for loan originated is steeper than for application denied.This shows that if an applicant applies for a loan of a specific amount their income should be proportionate to the amount loan applied.
plt.show()
plt.figure(figsize=(5, 3))

ax = sns.barplot(x="applicant_race_name_1", y="debt_to_income", hue="action_taken_name", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")], capsize=0.2)
# Looking at the plots it can be seen that applicants with asian thnicity have the highest debt to income ratio
#This is because as can be seen in the plot below the applicants with asian ethnicity applied for loans of higher value as compared to their income.
#Applicants with african american ethnicity show that their debt to income ratio is same for approval and denial of loans.
#This phenomenon can be explained by the count plot which shows that the major reason for application denial for applicants with black etnicity
#was their credit history which is not present in the dataset
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.figure(figsize=(5, 3))

ax = sns.barplot(x="applicant_race_name_1", y="loan_amount_000s",data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")], capsize=0.2)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')

plt.figure(figsize=(5, 3))

sns.countplot(data=loan[loan["applicant_race_name_1"] == "Black or African American"], x="denial_reason_name_1")
plt.show()
plt.figure(figsize=(5, 3))
ax = sns.barplot(x="action_taken_name", y="minority_population", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")], capsize=0.2)
#Here the bar graph shows that applicants living in higher minority population areas have a higher chance of being denied
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)

plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.show()
plt.figure(figsize=(5, 3))
ax = sns.barplot(x="applicant_race_name_1", y="minority_population", data=loan, capsize=0.2)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)

# sns.countplot(x="applicant_race_name_1",data=loan)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
#Plot shows that applicants with African american ethnicity live in live in areas with highest minority population,
#When it comes to minorities applicants with American indian ethnicity live in area with  least minority population minority population.
#Applicants with white ethnicity live overall live in areas with the least minority population

plt.show()
plt.figure(figsize=(5, 3))
ax = sns.countplot(x="applicant_sex_name", hue="action_taken_name", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")],
                   capsize=0.2)  # the count for male applicants is much larger than female applicants it remain to be seen if gender played a part in the acceptance or denial of loan
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="applicant_sex_name", y="applicant_income_000s", hue="action_taken_name", data=loan[
    ((loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")) & ((loan["applicant_sex_name"] == "Male") | (loan["applicant_sex_name"] == "Female"))],capsize=0.2)
#Here It can be seen that female applicants on average had less income as compared to their male counter parts.In the later graph we also see that female applicants apply for lower loans
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="applicant_sex_name", y="loan_amount_000s", hue="action_taken_name", data=loan[
    ((loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")) & ((loan["applicant_sex_name"] == "Male") | (loan["applicant_sex_name"] == "Female"))],capsize=0.2)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="applicant_sex_name", y="debt_to_income", hue="action_taken_name", data=loan[
    ((loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")) & ((loan["applicant_sex_name"] == "Male") | (loan[ "applicant_sex_name"] == "Female"))],
                 capsize=0.2)  #
#When considering debt to income ratio a correlation between loan approval and gender cannot be seen as both gender had their loans approved at relatively the same threshold
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.show()

plt.figure(figsize=(5, 3))
ax = sns.countplot(x="lien_status_name", hue="action_taken_name", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")],
                   capsize=0.2)  # the count for first lien is much larger than count for second lien

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="lien_status_name", y="debt_to_income", hue="action_taken_name", data=loan[
    ((loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution"))], capsize=0.2)  #
#Evidence from studies of mortgage loans suggest that borrowers with a higher debt-to-income ratio are more likely to run into trouble making monthly payments.
# The higher the debt to income ratio the more likely it becomes that loan might be denied.
# If a default of debt or a forced liquidation takes place, debt holders get paid in the following order:
#1. First-lien creditors
#2. Second-lien creditors
#This is what this plot shows as well that second lien loans are denied at a very low debt to income ratio compared to
#first lien loans.Considering the subordinated call on pledged collateral, secondary liens are more risky for lenders and investors.
# Since these loans are more risky, they typically have higher rates of interest and are subject to more stringent approval processes.
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)

plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.show()

plt.figure(figsize=(5, 3))
ax = sns.barplot(x="loan_purpose_name", y="applicant_income_000s", hue="action_taken_name", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")],
                 capsize=0.2)
plt.figure(figsize=(5, 3))
ax = sns.barplot(x="loan_purpose_name", y="loan_amount_000s", hue="action_taken_name", data=loan[
    (loan["action_taken_name"] == "Loan originated") | (
            loan["action_taken_name"] == "Application denied by financial institution")],
                 capsize=0.2)
plt.show()
#looking at these graphs it can be seen that on average it is relatively easier for applicants to get loans for home improvement as compared refinance to get loans to buy new homes
plt.figure(figsize=(5, 3))
ax = sns.barplot(x="owner_occupancy_name", y="applicant_income_000s", data=loan, capsize=0.2)  #
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
ax.set_ylim(0, 500)
# sns.countplot(x="applicant_race_name_1",data=loan)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.figure(figsize=(5, 3))
ax = sns.barplot(x="owner_occupancy_name", y="loan_amount_000s", data=loan, capsize=0.2)  #
for p in ax.patches:
    ax.annotate(f'\n{p.get_height().round(1)}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white',
                size=15)
ax.set_ylim(0, 500)
plt.setp(ax.get_xticklabels(), rotation=10, horizontalalignment='right')
plt.show()
#applicants where other used their home as their principal dwelling had shown to have higher income and have shown to request higher loans
#applicants where owner used their home as their principal dwelling had shown to have lower income and have shown to requestcomparitively lower loans

