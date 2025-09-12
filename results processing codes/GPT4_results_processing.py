import pandas as pd
import os
import shutil
import subprocess
from pathlib import Path

################# unzip ####################
# Diretorio onde os arquivos baixados estao localizados
diretorio = r"evaluation_spreadsheets\GPT-4\download_answers"

# Lista todos os arquivos no diretorio
arquivos = os.listdir(diretorio)

for arquivo in arquivos: #laco que passa por todos downloads zipados dos questionários
    # captura a extenao e usa o nome que vem do Forms para criar o nome que vai ser usado para identificar arquivos
    nome_arquivo, extensao = os.path.splitext(arquivo)
    nome_usavel = nome_arquivo.split()[0]
    print(nome_usavel) # soh para visualizar progresso e sucesso no terminal
    try:
        if extensao == ".zip":
            caminho_arquivo = os.path.join(diretorio+"_unzip", arquivo)
            shutil.unpack_archive(caminho_arquivo, diretorio)
    except:
        print("falha")
################# unzip ####################

################# settings  ####################
diretorio = r"evaluation_spreadsheets\GPT-4\download_answers_unzip" # directory whith the answers downloaded from Google Forms 
arquivos = os.listdir(diretorio) # List of files in the directory
################# settings ####################

df_12 = pd.DataFrame()

dic= {'qualidade da definição' : 'Acurácia - definição',
'nome escolhido' : 'Acurácia - nome',
'relação com seu(s) pai(s)' : 'Acurácia - pai(s)',
'agrega conhecimento' : 'Relevância',
'filhos geram cobertura suficiente' : 'Cobertura',
'posicionamento na ontologia' : 'Precisão - pos.',
'é, precisamente, um sub-conceito' : 'Precisão - subconceito',
'precisão das ligações' : 'Precisão - ligações',
'adequação do sinônimo' : 'Precisão - sinônimos'
}
            
c_Acuracia = ['Acurácia', 'Acurácia - definição', 'Acurácia - nome', 'Acurácia - pai(s)']
c_Cobertura = ['Cobertura']
#c_Precisao será definida depois pois pode variar
c_Relevância = ['Relevância']

for arquivo in arquivos:
    print(arquivo)    
    try:
        file_path = diretorio+"\\"+arquivo
        df = pd.read_csv(file_path)

        #começando da comluna a partir da qual queremos concatenar
        df_2 = df.drop(columns=['Carimbo de data/hora', 'Nome de usuário', "Nome (pode ser só o primeiro)"])

        df_nome = df[["Nome (pode ser só o primeiro)"]].rename(columns={"Nome (pode ser só o primeiro)":"Nome"})

        ######### Faz a quebra em linhas pra cada um dos n conceitos #########

        #no csv: cada linha tem as repostas de um dos x respondentes na forma:
        #    respondente1: todas_as_perguntas_coceito_1,..., todas_as_perguntas_coceito_n
        #    ...
        #    respondentex: todas_as_perguntas_coceito_1,..., todas_as_perguntas_coceito_n

        #df4: 
        #    respondente1: todas_as_perguntas_coceito_1
        #    ...
        #    respondentex: todas_as_perguntas_coceito_1
        #                                       ...
        #    respondente1: todas_as_perguntas_coceito_n
        #    ...
        #    respondentex: todas_as_perguntas_coceito_n
    
        df_3 = df_nome.copy()
        df_4 = pd.DataFrame()
        prior_concept = "placeholder"
        design = False
        
        for column in df_2:
        
            full_name = column.split("[")
            try:
                concept = full_name[0].split(":")[1].strip()
            except:
                concept = full_name[0].strip()

            if design or "(quantidade de níveis)" in concept:
                design = True            #necessário pra conseguir cortar colunas ao final da valiação de um conceito
                column_new = concept
            else:
                column_new = full_name[1].split("]")[0]
                try:
                    column_new = dic[column_new]
                except:
                    pass
        
            if prior_concept != "placeholder" and concept != prior_concept:
                df_3["Index"] =  prior_concept
                df_3.set_index("Index", inplace = True)
                df_4 = pd.concat([df_4, df_3])
                df_3 = df_nome.copy() #no final fica com as perguntads que sobram, que são as de design
            
            df_3 = pd.concat([df_3, df_2[column]], axis=1)
            df_3.rename(columns={column:column_new}, inplace = True)
            if design:
                prior_concept = "placeholder"
            else:
                prior_concept = concept
        
        df_5 = df_4.drop(columns=['Nome'])

        #não tem df_6 pq esse código foi adaptado do processo feito pras avaliações com outro formato de questionário, mas esse precisou de menos passos
        
        c_Precisao = ['Precisão', 'Precisão - pos.', 'Precisão - subconceito', 'Precisão - ligações']
        if 'Precisão - sinônimos' in df_5.columns:
            c_Precisao.append('Precisão - sinônimos')
        
        df_7 = df_4[["Nome"]].copy()
        for c in [c_Acuracia, c_Cobertura, c_Precisao, c_Relevância]:
            df_7[c[0]] = df_5[list(set(c))].mean(axis=1)

        df_8 = pd.DataFrame()
        df_8[["Nome", "Profundidade", "Quantidade", "Fora de escopo", "Avaliação geral","Comentários"]] = df_3
        df_8.set_index("Nome", inplace = True)

        df_9 = df_8[['Profundidade','Quantidade', 'Avaliação geral']].replace({'é bastante insuficiente' : 4.5,
        'é moderadamente insuficiente' : 2.5,'é suficiente' : 1,'é moderadamente exagerada' : 2.5,'é bastante exagerada' : 4.5,
        'não sei avaliar' : None}).replace({'Avaliação geral':{1 : 5,2 : 5,3 : 4,4 : 4,5 : 3,6 : 3,7 : 2,8 : 2,9 : 1,'10' : 1}})

        df_10 = df_7.groupby("Nome").mean()
        df_10["Desing de Informação"] = df_9.mean(axis=1)

        df_11 = df_10.mean()
        total = df_11.mean() #Nota final
        df_11 = df_11.to_frame().transpose()
        df_11['Total'] = total
        df_11["Initial_Concept"] = arquivo.split("_1")[-2].split()[-1]
        df_11.set_index("Initial_Concept", inplace = True)
        df_12 = pd.concat([df_12, df_11])
    except:
        print('Falha')