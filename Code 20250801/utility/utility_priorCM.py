import numpy as np
import pandas as pd

# # when deriving the proportion of monoculture and alternative rotation
def sd(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        result = 0
    return result

def convertM(string_matrix):
    # 去掉外层的方括号
    string_matrix = string_matrix.replace(" ", "")
    string_matrix = string_matrix.strip('[]')
    
    # 按行分割
    rows = string_matrix.split('],[')
    
    # 处理每行，转变成数字
    matrix = []
    for row in rows:
        # 移除每行的括号
        numbers = row.strip('[]')
        # 将每行按逗号分割，并转换为浮点型数字
        num_list = [float(num) for num in numbers.split(',')]
        matrix.append(num_list)
    
    # 转换为NumPy矩阵
    return np.array(matrix)

def get_rotation_Prop(regionName,yearLength=10,year=2021,cropTag=-1):

    xls = pd.ExcelFile("utility/prioCM.xlsx", engine='openpyxl')

    if cropTag !=-1:
        prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[5])
        prop = prop_data[(prop_data['year'] == year) & 
                            (prop_data['Croptype'] == cropTag)]
        prop = convertM(prop['Matrix'].values[0])

    else:
        if (yearLength < 10) & (year == 2021):
            yearLength = yearLength-1
            CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[2])
            prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[3])

            prop = prop_data[(prop_data['yearLength'] == yearLength) & 
                            (prop_data['region'] == regionName)]
            prop = convertM(prop['Matrix'].values[0])


        if (yearLength == 10) & (year >= 2021):
            CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[1])

            prop = prop_data[(prop_data['Year'] == year) & 
                            (prop_data['region'] == regionName)]
            prop = convertM(prop['Matrix'].values[0])

    propArray = np.array([[prop[0,0]/(prop[0,0]+prop[0,1]),prop[0,1]/(prop[0,0]+prop[0,1])],
                          [prop[1,0]/(prop[1,0]+prop[1,1]),prop[1,1]/(prop[1,0]+prop[1,1])]])
    return propArray

def get_rotation_size(regionName,yearLength=10,year=2021,cropTag=-1):

    xls = pd.ExcelFile("utility/prioCM.xlsx", engine='openpyxl')

    if (yearLength < 10) & (year == 2021):
        yearLength = yearLength-1
        CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[2])
        prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[3])
        prop = prop_data[(prop_data['yearLength'] == yearLength) & 
                        (prop_data['region'] == regionName)]
        prop = convertM(prop['Matrix'].values[0])


    if (yearLength == 10) & (year >= 2021):
        CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
        prop = prop_data[(prop_data['Year'] == year) & 
                        (prop_data['region'] == regionName)]
        prop = convertM(prop['Matrix'].values[0])

    return int(prop[1,:].max())

# Get the combined confusion matrix of trusted samples
def getPrioCM(regionName,yearLength=10,year=2021,cropTag=-1):

    xls = pd.ExcelFile("utility/prioCM.xlsx", engine='openpyxl')

    if cropTag !=-1:
        CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[4])

        prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[5])
        prop = prop_data[(prop_data['year'] == year) & 
                            (prop_data['Croptype'] == cropTag)]
        prop = convertM(prop['Matrix'].values[0])

        CM_mono = CM_data[(CM_data['year'] == year) & 
                    (CM_data['type'] == 'mono') & 
                    (CM_data['Croptype'] == cropTag)]
        CM_mono = convertM(CM_mono['predict_CM'].values[0])

        CM_alter = CM_data[(CM_data['year'] == year) & 
                        (CM_data['type'] == 'alter') & 
                        (CM_data['Croptype'] == cropTag)]
        CM_alter = convertM(CM_alter['predict_CM'].values[0])

        # return prop
    
    else:
        if (yearLength < 10) & ((year == 2021) | (year == 2022) | (year == 2024)):

            yearLength = yearLength-1 # 5 years indicates [1-4,5]

            CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[2])
            prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[3])

            if regionName == 'Qinghai':
                CM_mono = CM_data[(CM_data['yearLength'] == 3) & 
                    (CM_data['type'] == 'mono') & 
                    (CM_data['region'] == regionName)]
                CM_mono = convertM(CM_mono['predict_CM'].values[0])
                return CM_mono
            
            if regionName == 'NE':
                CM_mono = CM_data[(CM_data['yearLength'] == 3) & 
                    (CM_data['type'] == 'mono') & 
                    (CM_data['region'] == regionName)]
                CM_mono = convertM(CM_mono['predict_CM'].values[0])
                return CM_mono
            
            if regionName == 'WT':
                CM_alter = CM_data[(CM_data['yearLength'] == 4) & 
                    (CM_data['type'] == 'alter') & 
                    (CM_data['region'] == regionName)]
                CM_alter = convertM(CM_alter['predict_CM'].values[0])
                return CM_alter

            CM_mono = CM_data[(CM_data['yearLength'] == yearLength) & 
                    (CM_data['type'] == 'mono') & 
                    (CM_data['region'] == regionName)]
            CM_mono = convertM(CM_mono['predict_CM'].values[0])


            CM_alter = CM_data[(CM_data['yearLength'] == yearLength) & 
                            (CM_data['type'] == 'alter') & 
                            (CM_data['region'] == regionName)]
            
            CM_alter = convertM(CM_alter['predict_CM'].values[0])

            prop = prop_data[(prop_data['yearLength'] == yearLength) & 
                            (prop_data['region'] == regionName)]
            prop = convertM(prop['Matrix'].values[0])

        if (yearLength == 10) & (year >= 2021):
            CM_data = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            prop_data = pd.read_excel(xls, sheet_name=xls.sheet_names[1])

            CM_mono = CM_data[(CM_data['targetYear'] == year) & 
                    (CM_data['type'] == 'mono') & 
                    (CM_data['region'] == regionName)]
            CM_mono = convertM(CM_mono['predict_CM'].values[0])

            if (regionName == 'Qinghai')|(regionName == 'NE'):
                return CM_mono

            CM_alter = CM_data[(CM_data['targetYear'] == year) & 
                            (CM_data['type'] == 'alter') & 
                            (CM_data['region'] == regionName)]
            CM_alter = convertM(CM_alter['predict_CM'].values[0])

            prop = prop_data[(prop_data['Year'] == year) & 
                            (prop_data['region'] == regionName)]
            prop = convertM(prop['Matrix'].values[0])

    prop = np.array([[prop[0,0]/(prop[0,0]+prop[0,1]),prop[0,1]/(prop[0,0]+prop[0,1])],
                          [prop[1,0]/(prop[1,0]+prop[1,1]),prop[1,1]/(prop[1,0]+prop[1,1])]])

    tN = 100000

    p1 = prop[0, 0]
    p2 = prop[0, 1]

    p1_ = prop[1, 0]
    p2_ = prop[1, 1]

    CM_mono_prop = np.array([
        [sd(CM_mono[0, 0], (CM_mono[0, 0] + CM_mono[0, 1])), sd(CM_mono[0, 1], (CM_mono[0, 0] + CM_mono[0, 1]))],
        [sd(CM_mono[1, 0], (CM_mono[1, 0] + CM_mono[1, 1])), sd(CM_mono[1, 1], (CM_mono[1, 0] + CM_mono[1, 1]))]
    ])
    CM_mono_prop = np.nan_to_num(CM_mono_prop, nan=0)
    
    CM_alter_prop = np.array([
        [sd(CM_alter[0, 0], (CM_alter[0, 0] + CM_alter[0, 1])), sd(CM_alter[0, 1], (CM_alter[0, 0] + CM_alter[0, 1]))],
        [sd(CM_alter[1, 0], (CM_alter[1, 0] + CM_alter[1, 1])), sd(CM_alter[1, 1], (CM_alter[1, 0] + CM_alter[1, 1]))]
    ])
    CM_alter_prop = np.nan_to_num(CM_alter_prop, nan=0)

    CM_mono_new = np.array([
        [tN * p1, tN * p1],
        [tN * p1_, tN * p1_]
    ]) * CM_mono_prop
    
    CM_alter_new = np.array([
        [tN * p2, tN * p2],
        [tN * p2_, tN * p2_]
    ]) * CM_alter_prop

    CM = CM_mono_new + CM_alter_new

    return CM
