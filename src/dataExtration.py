import numpy as np
import pandas as pd

with open("data/rowdata/h80_lvl_01.txt") as fil:
  lines1= fil.readlines()

df1=[]
for i in range(len(lines1)):
  v1=lines1[i][0:35]
  unique_no=v1.replace(" ","0")

  household_size=lines1[i][53:55]
  HUCE=lines1[i][107:115]
  sector=lines1[i][11:12]
  state=lines1[i][12:14]
  religion=lines1[i][55:56]
  social_group=lines1[i][56:57]
  household_type=lines1[i][57:58]
  mult1=lines1[i][115:127]


  df1.append([unique_no,household_size,HUCE,sector,state,religion,social_group,household_type,mult1])


DF1=pd.DataFrame(df1)
DF1.columns=["unique_no","h_size","HUCE","sector","state","religion","social_group","h_type","mult1"]


cols=DF1.columns


#all columns are in str convert them in int

for i in cols[1:]:
  DF1[i]=DF1[i].astype(str) # Explicitly convert to string first
  DF1[i]=DF1[i].str.strip()
  DF1[i]=DF1[i].str.replace(" ","0")
  DF1[i]=DF1[i].replace("", "0")
  DF1[i]=DF1[i].astype(int)


#2nd file
with open("data/rowdata/h80_lvl_02.txt") as fil:
  lines2= fil.readlines()

df2=[]
for i in range(len(lines2)):
  v1=lines2[i][0:35]
  unique_no=v1.replace(" ","0")
  relation_with_Head=lines2[i][39:40]
  Gender=lines2[i][40:41]
  Age=lines2[i][41:44]
  H_education=lines2[i][45:47]
  Whether_hosp=lines2[i][48:49]
  Whether_last_15_days_dis_yORn=lines2[i][57:58]
  insurance_covered_yORn = lines2[i][59:61]
  mult2=lines2[i][61:73]


  df2.append([unique_no,relation_with_Head,Gender,Age,H_education,Whether_hosp,Whether_last_15_days_dis_yORn,insurance_covered_yORn,mult2])


import pandas as pd
DF2=pd.DataFrame(df2)
DF2.columns=["unique_no","relation_with_Head","Gender","Age","H_education","Whether_hosp","Whether_last_15_days_dis_yORn","insurance_covered_yORn","mult2"]

cols=DF2.columns
for i in cols[1:]:
  DF2[i]=DF2[i].astype(str) # Explicitly convert to string first
  DF2[i]=DF2[i].str.strip()
  DF2[i]=DF2[i].str.replace(" ","0")
  DF2[i]=DF2[i].replace("", "0")
  DF2[i]=DF2[i].astype(int)



#4th file
with open("data/rowdata/h80_lvl_04.txt") as fil:
  lines4= fil.readlines()

df4=[]
for i in range(len(lines4)):
  v1=lines4[i][0:35]
  unique_no=v1.replace(" ","0")

  Nature_Dis=lines4[i][47:49]
  type_Hopital=lines4[i][50:51]

  total_expences=lines4[i][147:155]
  total_reimbursed=lines4[i][155:163]
  mejor_source_for_finance=lines4[i][163:164]
  loss_of_household_income=lines4[i][167:175]
  mult4=lines4[i][175:187]
  col=[unique_no,Nature_Dis,type_Hopital,total_expences,total_reimbursed,mejor_source_for_finance,mult4]
  df4.append(col)


import pandas as pd
DF4=pd.DataFrame(df4)
cols="unique_no,Nature_Dis,type_Hopital,total_expences,total_reimbursed,mejor_source_for_finance,mult4".split(",")

DF4.columns=cols
for i in cols[1:]:
  DF4[i]=DF4[i].astype(str) # Explicitly convert to string first
  DF4[i]=DF4[i].str.strip()
  DF4[i]=DF4[i].str.replace(" ","0")
  DF4[i]=DF4[i].replace("", "0")
  DF4[i]=DF4[i].astype(int)



#5th file
with open("data/rowdata/h80_lvl_05.txt") as fil:
  lines5= fil.readlines()

df5=[]
for i in range(len(lines5)):
  v1=lines5[i][0:35]
  unique_no=v1.replace(" ","0")

  Nature_Dis_lst_15_days=lines5[i][47:49]
  level_of_care_last_15days =lines5[i][61:62]
  total_expences_last_15days_Rs=lines5[i][135:143]
  total_reimbursed_last_15days_Rs=lines5[i][143:151]

  mejor_source_for_finance_last_15days=lines5[i][151:152]
  mult5=lines5[i][163:175]


  col=[unique_no,Nature_Dis_lst_15_days,level_of_care_last_15days,total_expences_last_15days_Rs,total_reimbursed_last_15days_Rs,mejor_source_for_finance_last_15days,mult5]
  df5.append(col)


cols=["unique_no","Nature_Dis_lst_15_days","level_of_care_last_15days","total_expences_last_15days_Rs","total_reimbursed_last_15days_Rs","mejor_source_for_finance_last_15days","mult5"]

# cols



import pandas as pd
DF5=pd.DataFrame(df5)
DF5.columns=cols

for i in cols[1:]:
  DF5[i]=DF5[i].astype(str) # Explicitly convert to string first
  DF5[i]=DF5[i].str.strip()
  DF5[i]=DF5[i].str.replace(" ","0")
  DF5[i]=DF5[i].replace("", "0")
  DF5[i]=DF5[i].astype(int)

print("All data extracted succesfully ::::::")


DF1.to_csv("data/processed_data/DF1.csv", index=False)
DF2.to_csv("data/processed_data/DF2.csv", index=False)
DF4.to_csv("data/processed_data/DF4.csv", index=False)
DF5.to_csv("data/processed_data/DF5.csv", index=False)