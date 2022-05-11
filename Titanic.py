# =========================================== INSPER - CIENCIA DE DADOS E INTELIGENCIA ARTIFICIAL ==========================================
# ------------------------------------------------------------- KAGGLE | TITANIC -----------------------------------------------------------



## Objetivo do Trabalho
# Link da competição do Kaggle > https://www.kaggle.com/competitions/titanic/overview
# Escrever um código que consiga prever se uma pessoa sobreviveu ou não ao acidente do Titanic, a partir de suas caracteristicas 

## Alunos
# Julia Wolf Mazzuia
# Gustavo Molina
# Kevin Saraiva
# Lucas Cury

## Bibliografia e Inspirações
# Toti Cavalcanti > https://www.youtube.com/watch?v=UyPZnO4euR8&t=3s
# Ken Jee > https://www.kaggle.com/code/kenjee/titanic-project-example/notebook


## Challenge
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
#
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. 
# Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?”
# using passenger data (ie name, age, gender, socio-economic class, etc).


## What Data Will I Use in This Competition?
# In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic 
# class, etc. One dataset is titled `train.csv` and the other is titled `test.csv`.
# 
# Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, will reveal whether they 
# survived or not, also known as the “ground truth”.
#
# The `test.csv` dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict 
# these outcomes.
# 
# Using the patterns you find in the train.csv data, predict whether the other 418 passengers on board (found in test.csv) survived.
# Check out the “Data” tab to explore the datasets even further. Once you feel you’ve created a competitive model, submit it to Kaggle to 
# see where your model stands on our leaderboard against other Kagglers.


## Lista de transformações necessárias
# 1) Coluna Survived com missing values
# Foi necessário criar uma coluna Survived e preencher com missing values para juntar as bases train e test
#
# 2) Coluna istrain
# Foi necessário criar uma coluna istrain para indentificar na base de dados que junta train e test, se é ou não originalmente da train
# 
# 3) Escolher o que fazer com missing values
# 
#
# 4) Variáveis Ticket e Cabin se beneficiariam de um tratamento
# A maneira que recebemos as variáveis não é muito insightful. Agrupa-los por algum critério deve ajudar a tomar decisões melhores



# --------------------------------------------------------------- 01. INTRO ----------------------------------------------------------------



# Importa bibliotecas
from http.client import MULTI_STATUS
import os                                                                                                                                   # Biblioteca para manuseio de arquivos
import numpy as np                                                                                                                          # Biblioteca para manuseio de dados em matriz e distribuições
import pandas as pd                                                                                                                         # Biblioteca para manuseio de dados em DataFrame
from dfply import *                                                                                                                         # Biblioteca para manuseio de dados semelhante ao R
from os.path import exists                                                                                                                  # Biblioteca para manuseio de arquivos
import progressbar as pb                                                                                                                    # Biblioteca que permite ver o progresso do código em barra
import time                                                                                                                                 # Biblioteca que permite realizar operações relacionadas com tempo 
from datetime import timedelta                                                                                                              # Biblioteca para calcular duração de trechos do código
from datetime import datetime                                                                                                               # Biblioteca para dizer data de hoje
import seaborn as sns                                                                                                                       # Biblioteca para plotar gráficos
import matplotlib.pyplot as plt                                                                                                             # Biblioteca para plotar gráficos
import missingno as mn                                                                                                                      # Biblioteca para ver missing values de uma maneira interessante
from pycaret.utils import enable_colab                                                                                                      # Modo que habilita o uso do Carte no Colab do Google
from pycaret.classification import *                                                                                                        # Biblioetca que permite várias análises de classificação
import pprint as pp                                                                                                                         # Bbiblioteca que organiza print
from sklearn.ensemble import RandomForestClassifier                                                                                         # Random Forest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Habilita um modo de funcionamento do py.caret caso use o colab
# enable_colab()

# Clock inicio código
Start_Time = time.monotonic()

# Define o diretório onde está salvo os arquivos que serão utilizados
wdir = os.getcwd()                                                                                                                          # Guarda a localização do diretório do arquivo
wdir = wdir.replace("\\", '/')                                                                                                              # Troca o padrão de localização da Microsoft Windows para o padrão universal
os.chdir(wdir)                                                                                                                              # Define esse como o diretório padrão para esse algoritimo

# Cria um caminho para puxar os dados brutos e outro para o armazenamento dos resultados
inputs_path = "/01. Inputs/"
result_path = "/02. Result/"

# Indica etapa do processo como concluida
print("01. Intro | OK")



# -------------------------------------------------------------- 02. COCKPIT ---------------------------------------------------------------



# Hard Inputs
Version_save = datetime.now().strftime("%Y-%m-%d--%Hh%M")                                                                                   # Nome da versão de cópia do arquivo
train_file = wdir + inputs_path + "train.csv"                                                                                               # Nome do arquivo de de treino
test_file = wdir + inputs_path + "test.csv"                                                                                                 # Nome do arquivo de teste
Titanic_copy = wdir + result_path + "Titanic_" + Version_save + ".xlsx"                                                                     # Nome do arquivo de cópia de segurança
Titanic_file =  wdir + inputs_path + "Titanic.xlsx"                                                                                         # Nome do arquivo de resultado
results_file =  wdir + inputs_path + "gender_submission.csv"


# Indica etapa do processo como concluida
print("02. Cockpit | OK")



# ----------------------------------------------------------- 03. PREPARAR DADOS -----------------------------------------------------------



# Clock inicio etapa
start_time = time.monotonic()

# Cria a base de dados que contem o conjunto de treino e faz algumas transformações
train = (pd.read_csv(train_file) >>                                                                                                         # Le o arquivo de csv
        mutate(Train = 1))                                                                                                                  # Cria uma coluna que indentifica se o arquivo é de treino

# Cria base de dados que contem o conjunto de teste e faz algumas transformações
test = (pd.read_csv(test_file) >>                                                                                                           # Le o arquivo de csv
        mutate(Train = 0))                                                                                                               # Cria uma coluna que indentifica se o arquivo é de treino

test_results_column = pd.read_csv(results_file)['Survived']

# Cria uma base de dados que junta as base de dados e faz transformações
full = (train >>
        bind_rows(test,                                                                                                                     # Junta a base de dados train com a test
            join = "outer") >>                                                                                                              # Método outer
        mutate(NameComplete = X.Name,                                                                                                       # Cria uma nova coluna como nome completo, já que vamos transformar a coluna nome em nome simples
            Cabin_mult = X.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(" "))),                                                   # Cria uma coluna dizendo se havia dado de Cabin ou não 
            Cabin_adv = X.Cabin.apply(lambda x: str(x)[0]),                                                                                 # Captura a primeira letra da coluna que indentifica a cabine
            Num_ticket = X.Ticket.apply(lambda x: 1 if x.isnumeric() else 0),                                                               # Verifica se há trechos somente numéricos no ticket
            Letter_ticket = X.Ticket.apply(lambda x: "".join(x.split(" ")[:-1])                                                             # Captura a informação importante do Ticket
                .replace(".","")
                .replace("/","")
                .lower() if len(x.split(" ")[:-1]) >0 else 0)) >>                                                                           #                                          
        separate(X.Name,                                                                                                                    # Separa sobrenome de nome
            ["Surname", "Name"],                                                                                                            # Nomeia as colunas que receberão as informações
            sep = ",",                                                                                                                      # Por meio de virgulas
            remove = False) >>                                                                                                              # Define que a coluna original (Name), não será deletada
        separate(X.Name,                                                                                                                    # Separa título de nome
            ["Title"],                                                                                                                      # Nomeia a coluna que receberá a informação
            sep = ". ",                                                                                                                     # Por meio de ponto
            remove = False))                                                                                                                # Define que a coluna original (Name), não será deletada

# Retorna os dados de train, mas com as transformações feitas em full
train = (full >>                                                                                                                            # Substitui a base train pela full
        mask(X.Train == 1))                                                                                                                 # Filtrando somente os dados que eram originalmente da train

# Retorna os dados de test, mas com as transformações feitas em full
test = (full >>                                                                                                                             # Substitui a base test pela full
        mask(X.Train == 0))                                                                                                                 # Filtrando somente os dados que eram originalmente da test

# Cria uma seleção base de dados train somente com as colunas númericas
train_num = (train >>
            select(X.Age,
                X.SibSp,
                X.Parch,
                X.Fare))

# Cria uma seleção base de dados train somente com as colunas categóricas
train_cat = (train >>
            select(X.Survived,
                X.Pclass,
                X.Sex,
                X.Ticket,
                X.Cabin,
                X.Embarked))

# Clock fim etapa
end_time = time.monotonic()
print("03. Preparar Dados | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")



# --------------------------------------------------------- 04. ANÁLISE DESCRITIVA ---------------------------------------------------------



# Clock inicio etapa
start_time = time.monotonic()

# Mostra uma tabela com algumas estatíticas básicas sobre as colunas de cada uma das base de dados
train.describe()
test.describe()
full.describe()


## 4.1 Análise descritiva variáveis númericas da base train
# Gráficos e tabelas que ajudam a entender as variáveis numéricas da base de dados train

# Plota histograma das variáveis categóricas
# for g in train_num.columns:
#     plt.hist(train_num[g])                                                                                                                  # Prepara o histograma da variável
#     plt.title(g)                                                                                                                            # Define que o titulo do histograma é o nome da variável 
#     plt.show()                                                                                                                              # Plota de fato os gráficos

# Cria tabela de correlação entre as variáveis em heatmap
# print(train_num.corr())                                                                                                                     # Mostra tabela com valores das correlações entre as variáveis
sns.heatmap(train_num.corr())                                                                                                               # Plota tabela com usando cores em gradient para definir grau de correlação

# Cria tabela da média de cada variável numérica caso a pessoa tenha sobrevivido ou não
surv_table_num = (train >>
    group_by(X.Survived) >>
    summarize(Mean_Age = mean(X.Age),
        Mean_SibSp = mean(X.SibSp),
        Mean_Parch = mean(X.Parch),
        Mean_Fare = mean(X.Fare)))
# print(surv_table_num)


## 4.2 Análise descritiva variáveis categóricas da base train
# Gráficos e tabelas que ajudam a entender as variáveis categoricas da base de dados train

# Cria gráficos que comparam a contagem de dados váildos por categoria
# for c in train_cat.columns:
#     sns.barplot(train_cat[c].value_counts().index,
#         train_cat[c].value_counts()).set_title(c)
#     plt.show()\

# Cria tabela da média de cada variável numérica caso a pessoa tenha sobrevivido ou não
surv_table_Pclass = pd.pivot_table(train, 
                    index = "Survived", 
                    columns = "Pclass", 
                    values = "Ticket" ,
                    aggfunc ="count")

surv_table_Sex = pd.pivot_table(train, 
                    index = "Survived", 
                    columns = "Sex", 
                    values = "Ticket" ,
                    aggfunc ="count")

surv_table_Embarked = pd.pivot_table(train, 
                    index = "Survived", 
                    columns = "Embarked", 
                    values = "Ticket" ,
                    aggfunc ="count")

surv_table_Title = pd.pivot_table(train, 
                    index = "Survived", 
                    columns = "Title", 
                    values = "Ticket" ,
                    aggfunc ="count")

# Printa as tabelas com as agregações feitas anteriormente
# print(surv_table_Pclass)
# print()
# print(surv_table_Sex)
# print()
# print(surv_table_Embarked)
# print()
# print(surv_table_Title)

# Contagem de pessoas por título
surv_table_title = (full >>
                    mask(X.Train == 1) >>
                    group_by(X.Survived) >>
                    summarize(Count = n(X.Title)))
                    


## 4.3 Feature Engeneering
# Propor transformações nos dados para manter a integridade da análise principal

# Mostra Missing values de cada uma das base de dados -> Embarked, Cabin e Age precisam sofrer algum tipo de ajuste por terem NaN em train
mn.matrix(train)
mn.matrix(test)
mn.matrix(full)


## 4.3.1 Embarked
# Como são somente 2 observações com problema em Embarked na base train, a decisão foi dropar essas observações

# Retorna a base full sem as observações que tinham Embarked como missing value
full.dropna(subset = ["Embarked"], inplace = True)


## 4.3.2 Fare
# Somente um dado do conjunto test não possui dado em Fare. Não podemos dropar essa informação, já que faz parte da resposta final, então
# para possibilitar o uso dessa coluna em modelos preditivos vamos substituir pela mediana da variável

# Cacula mediana das idades da base de dados que tem tanto os dados teste quanto train
median_fare = median(full.Fare)

# Substitui os missing values de idade pela mediana
full = (full >>
        mutate(Fare = X.Fare.fillna(median_fare)))


## 4.3.3 Age
# Uma quantidade consideravel dos dados não tem idade. Uma forma simples de não precisar dropar observações sem idade é substituir pela 
# mediana das idades 

# Cacula mediana das idades da base de dados que tem tanto os dados teste quanto train
median_age = median(full.Age)

# Substitui os missing values de idade pela mediana
full = (full >>
        mutate(Age = X.Age.fillna(median_age)))

# Realiza as transformações de normalização - tiveram que ser feitas separadas pelo Python não ter delayed interpretation como o R
full = (full >>
        mutate(Norm_sibsp = np.log(full.SibSp + 1),                                                                                         # Cria uma coluna normalizada da quantidade de irmãos ou significant other a bordo
            Norm_parch = np.log(full.Parch + 1),                                                                                            # Cria uma nova coluna normalizada da quantidade de filhos ou país a bordo
            Norm_fare = np.log(full.Fare + 1)))                                                                                             # Cria uma coluna normalizada do preço pago para a viagem

# Função que cria coluna com valor de linha contendo 1 como default
def create_title_column(row, column):
    row[(row[column]).title()] = 1
    
    return row

# Função que substitui os valores de embarked por numericos
def assign_values(row, dict_labels):
    row['Embarked'] = dict_labels[row['Embarked']]
    
    return row

# Aqui aplicamos a função previamente mencionada (assign_values) no dataframe e substituimos por n.a os outros valores achados como unicos na coluna
for column in ['Title', 'Cabin_adv', 'Sex']:
    full = full.apply(lambda row: create_title_column(row, column), axis = 1)

    for title in list(full[column].unique()):
        full[title.title()] = full[title.title()].fillna(0)

# Criamos um dict de de(string)-para(numerico) para as variaveis em Embarked
dict_labels = {
    full['Embarked'].unique()[iter_label] : iter_label for iter_label in range(len(list(full['Embarked'].unique())))
}

# Aplicamos a função assign_values
full = full.apply(lambda row: assign_values(row, dict_labels), axis = 1)

# Dropamos as colunas não numericas
for drop_col in ['Title', 'Cabin_adv', 'Sex', 'Cabin', 'Cabin_mult', 'Letter_ticket', 'Name', 'NameComplete', 'Surname', 'Ticket']:
    full = full.drop(drop_col, 1)

## 4.4 Finalizar trabalho na base full
# Salva um conjunto de dummies
# Dummies = pd.get_dummies(full
#     [["Pclass", "Sex", "Age", "SibSp", 
#     "Parch", "Norm_fare","Embarked","Cabin_adv",
#     "Cabin_mult", "Num_ticket", "Title", "Train"]])

# Faz uma última checagem se há algo que ainda precise de algum tipo de tratamento
mn.matrix(full)

# Volta a separar entre base treino...
train = (full >>
        mask(X.Train == 1))

# ...e base de teste
test = (full >>
        mask(X.Train == 0))

test_survived = test['Survived']
test = test.drop('Survived', axis = 1)

# Clock fim etapa
end_time = time.monotonic()
print("04. Análise Descritiva | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")

# --------------------------------------------------------------- 05. MODELO ---------------------------------------------------------------

# Clock inicio etapa
start_time = time.monotonic()

#criamos uma copia do treino
cp_train = train.copy()

# Aqui é feito o random forest para vermos quais as variaveis mais importantes do modelo
rf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
survived = cp_train['Survived']
cp_train = cp_train.drop('Survived',axis = 1)
rf = rf.fit(cp_train, survived)

# Criamos um dataframe para alocar os resultados
rf_df = pd.DataFrame()

# Criamos as colunas Count que possui a relevancia de cada variavel alvo e Values que possui o nome de cada coluna
rf_df['Count'] = rf.feature_importances_
rf_df['Values'] = cp_train.columns

# Fazemos um sort para vermos as mais relevantes e setamos o index do dataframe para o nome da propria coluna
rf_df = rf_df.sort_values(by=['Count'], ascending = True)
rf_df = rf_df.set_index('Values')

rf_df.plot(kind = 'barh')

# Aqui sera abrigado o resultado da cross validation
dict_scores = {}

# Inicializamos todos os modelos e colocamos cada um deles dentro de uma lista
models_list = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(), KNeighborsClassifier()]

# Validação cruzada entre diferentes tipos de modelos
for model in models_list:
    result = np.mean(cross_val_score(model, cp_train, survived, cv = 4, scoring='accuracy'))
    print('* ----------------------------------------------------------------------------------- *')
    print('  Validação cruzada de: {}'.format(model.__class__))
    print('  Cross Score: {}'.format(result))
    print('* ----------------------------------------------------------------------------------- *')
    print()
    dict_scores['{}'.format(model)] = result

print("O modelo escolhido pelo cross validation para ser utilizado foi o : {}".format(max(dict_scores, key = dict_scores.get)))
print("Começando tuning do modelo..")

# Selecionamos o modelo que obteve o melhor resultado no cross score
final_model = max(dict_scores, key = dict_scores.get)

# Aqui iremos fazer um hyperparameter tuning para os modelos que em media obtiveram maior acertividade (Regressão Linear e Random Forest)
if final_model == 'LogisticRegression()':
    final_model = models_list[0]
    
    # Aqui temos os tipos de solver sem ser sag e saga para distribuições lineares que iremos variar em nosso gri search
    solvers = ['newton-cg', 'lbfgs', 'liblinear']

    # Penalty sera unica (apesar de existir outras opções, o l2 abrange todos os tipos de solver para este modelo)
    penalty = ['l2']

    # Aqui mostramos as opções de C (cross-product)
    c_values = [100, 10, 1.0, 0.1, 0.01]

    # Numero maximo de iterações feito de forma ao modelo não dar erro de numero maximo de iterações alcançado
    max_iter = [6000]

    # Aqui é feito o dict com as configurações previamente citadas
    grid = dict(solver=solvers,penalty=penalty,C=c_values, max_iter = max_iter)

    # Inicializamos um RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=4, random_state=1)
    
    # Aqui jogamos o RepeatedStratifiedKFold junto com nosso dict de configurações para o modelo e começamos a variar suas entradas
    grid_search = GridSearchCV(estimator=final_model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(cp_train, survived)

    print("Melhor acerto: {} utilizando {}".format(grid_result.best_score_, grid_result.best_params_))

    # É feito o predict
    predict = grid_result.predict(test)

    # Aqui é feito um class report e uma matriz confusão para sabermos quão bem nosso modelo desempenhou
    print("Clas Report Output: ")
    clas_report = classification_report(list(test_results_column), predict)
    conf_matrix = confusion_matrix(list(test_results_column),predict)

    pp.pprint(clas_report)
    print()
    print("Confusion Matrix Output: ")
    pp.pprint(conf_matrix)
    print()

    test['Survived'] = list(predict)

if final_model == 'RandomForestClassifier()':
    final_model = models_list[1]
    
    # Criamos um vetor que vai de 100 a 1500 variando de 10 em 10 para ser utilizado no campo estimators
    estimators = [int(x) for x in np.linspace(start = 100,stop = 1500,num = 10)]

    # Aqui mostramos as features utilziadas no modelo que irao variar em nosso RandomizedSearch
    features = ['auto','sqrt']

    # Criamos um vetor que vai de 5 a 120 variando de 10 em 10
    depth = [int(x) for x in np.linspace(start = 5,stop = 120,num = 10)]

    # Lista que mostra valor de split maximo para a arvore
    split = [2, 5, 10]

    # Lista que mostra numero de folhas maximo para a arvore
    leafs = [1,2,4]

    # Lista para variar modelo com e sem bootstrap
    bootstrap = [True, False]

    # Grid com configurações previamente criadas para ser utilizada como parametro do modelo no RandomizedSearchCV
    grid = {
        'n_estimators' : estimators,
        'max_features' : features,
        'max_depth' : depth,
        'min_samples_split' : split,
        'min_samples_leaf' : leafs,
        'bootstrap' : bootstrap
    }

    # Aqui é feita a variação dos parametros do modelo de forma a ser encontrados os parametros que geram a melhor resposta 
    rf_grid = RandomizedSearchCV(estimator = final_model, param_distributions = grid, n_iter = 100, cv = 4, verbose = 2, random_state = 1, n_jobs = -1)
    rf_grid.fit(cp_train, survived)

    print("Melhor acerto: {} usando {}".format(rf_grid.best_score_, rf_grid.best_params_))

    # Feito o predict na base de teste
    predict = rf_grid.predict(test)

    # Aqui é feito um class report e uma matriz confusão para sabermos quão bem nosso modelo desempenhou
    print("Clas Report Output: ") 
    clas_report = classification_report(list(test_results_column), predict)
    conf_matrix = confusion_matrix(list(test_results_column),predict)

    pp.pprint(clas_report)
    print()
    print("Confusion Matrix Output: ")
    pp.pprint(conf_matrix)
    print()

    test['Survived'] = list(predict)

# Clock fim etapa
end_time = time.monotonic()
print("05. Modelo | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")

# # ------------------------------------------------------------ 99. SALVA ARQUIVO -----------------------------------------------------------



# Clock inicio etapa
start_time = time.monotonic()

# Salva o progresso
with pd.ExcelWriter(Titanic_file) as writer:  
    train.to_excel(writer, sheet_name = "train")
    test.to_excel(writer, sheet_name = "test")
    full.to_excel(writer, sheet_name = "full")
    train_num.corr().to_excel(writer, sheet_name = "train_num_corr")
    surv_table_num.to_excel(writer, sheet_name = "surv_table_num")

# Salva o progresso em um arquivo de segurança
with pd.ExcelWriter(Titanic_copy) as writer:  
    train.to_excel(writer, sheet_name = "train")
    test.to_excel(writer, sheet_name = "test")
    full.to_excel(writer, sheet_name = "full")
    train_num.corr().to_excel(writer, sheet_name = "train_num_corr")
    surv_table_num.to_excel(writer, sheet_name = "surv_table_num")

# Clock fim etapa
end_time = time.monotonic()
print("99. Salva arquivo | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")

# Clock fim código
End_Time = time.monotonic()
print("--------------------------")
print("Fim do Código")
print(f"Code Duration: {timedelta(seconds = End_Time - Start_Time)}")
print("--------------------------")


# --------------------------------------------------------------- 00. ESBOÇO ---------------------------------------------------------------

