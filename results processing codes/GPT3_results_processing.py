import pandas as pd
import os
import shutil
import subprocess
from pathlib import Path

################# unzip ####################
# Diretorio onde os arquivos baixados estao localizados
diretorio = r"evaluation_spreadsheets\GPT-3.5\download_answers"

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
diretorio = r"evaluation_spreadsheets\GPT-3.5\download_answers_unzip" # directory whith the answers downloaded from Google Forms 
arquivos = os.listdir(diretorio) # List of files in the directory
################# settings ####################

df_12 = pd.DataFrame()

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
        aprs = [0,0,0,0] #contador para passagem pelos múltiplos pais pelas questões de Acurácia, Precisão, Relevância, e por possíveis múltiplos sinônimos
        cont = 0

        c_Acuracia = ["Acurácia – nome escolhido","Acurácia – qualidade da definição"]
        c_Cobertura = ["Cobertura", "Cobertura – conceitos folha"]
        c_Precisao = ["Precisão – posicionamento", "Precisão das ligações"]
        c_Relevância = []

        for column in df_2:
            column_new = column

            df_3 = pd.concat([df_3, df_2[column]], axis=1)

            # ver arquivo ideia _formato dos resultados.ods para entender esse mapeamento
            if "[o nome escolhido para o conceito é adequado]" in column:
                column_new = "Acurácia – nome escolhido"
            elif "[a definição do conceito está correta]" in column:
                column_new = "Acurácia – qualidade da definição"
            elif "relacionado ao seu conceito pai" in column:
                aprs[0] += 1
                column_new = "Acurácia – relação com seu pai "+str(aprs[0])
                c_Acuracia.append(column_new)
            elif "suficiente para destrinchar" in column:
                column_new = "Cobertura"
            elif "o quão interessante seria expandi-lo" in column:
                column_new = "Cobertura – conceitos folha"
            elif "conceito possui o sinônimo" in column:
                aprs[3] += 1
                column_new = "Precisão – adequação do sinônimo "+str(aprs[3])
                c_Precisao.append(column_new)
            elif "lugar para posicionar" in column:
                column_new = "Precisão – posicionamento"
            elif "em relação ao seu conceito-pai" in column:
                aprs[1] += 1
                column_new = "Precisão – tipo de relação "+str(aprs[1])
                c_Precisao.append(column_new)
            elif '"Outro"' in column:
                column_new = "Precisão – tipo de relação – Outros"
            elif "deveria estar ligado como filho" in column:
                column_new = "Precisão das ligações"
            elif "agrega conhecimento" in column:
                aprs[2] += 1
                column_new = "Relevância "+str(aprs[2])
                c_Relevância.append(column_new)

            df_3.rename(columns={column:column_new}, inplace = True)

            if "sse conceito deveria estar ligado " in column:
                cont += 1
                df_3["Index"] = cont
                df_3.set_index("Index", inplace = True)
                df_4 = pd.concat([df_4, df_3])
                df_3 = df_nome.copy() # ele é renovado ao final de cada conceito. Quando acabam os conceitos ele faz mais uma passada e fica só com as perguntads de design
                #if aprs[3] > 1: print(cont, aprs[3]) #era só pra ver se tinha algum caso de mais de um sinônimo pra um conceito
                aprs = [0,0,0,0]

        df_4 = df_4.reindex(sorted(df_4.columns), axis=1)

        df_5 = df_4.drop(columns=['Nome','Precisão – tipo de relação – Outros']).replace(["não sei avaliar", "Outro"], None).replace({'Cobertura' : {"o conceito não precisava ter sido expandido" : None},
        'Cobertura – conceitos folha' : {'Muito' : 5,'Moderado' : 4,'Neutro' : 3,'Seria um pouco desnecessário' : 2,'Seria totalmente desnecessário' : 1},
        'Precisão – adequação do sinônimo 1' : {'os conceitos podem tranquilamente ser usados de forma intercambiável (sinônimos ideais)' : 1,'há similaridade mas não são sinônimos ideais' : 2,'um é uma tradução correta do outro' : 3,'um é uma tradução inadequada do outro' : 4,'um é plural do outro' : 3,'os conceitos diferem de forma significativa' : 5},
        'Precisão – posicionamento' : {"esse conceito não deveria fazer parte da ontologia" : None},
        'Precisão – tipo de relação 1': {'uma subcategoria/sub-conceito (assim como Bolo estaria para Doce. Todo bolo é um doce, mas não necessariamente todo doce é um bolo.)' : 1,'uma instância (assim como Lagoa da Pampulha estaria para Lagoa)' : 2,'uma parte (assim como Cotovelo estaria para Braço)' : 2,'um irmão (deveriam estar ligados a um mesmo conceito-pai)' : 3,'um sinônimo (toda instância dessa categoria está também na outra, e vice-versa. Ex: Prédio e Edifício (no uso coloquial))' : 3,'a relação está invertida (esse conceito é que deveria ser o conceito-pai do outro)' : 4},
        'Precisão das ligações' : {0 : 1,1 : 2,2 : 3,3 : 4,'4 ou mais' : 5}}
        ).replace({'uma subcategoria/sub-conceito' : 1, 'uma subcategoria' : 1,'uma instância' : 2,'uma parte' : 2,'um irmão' : 3,'um sinônimo' : 3,'a relação está invertida' : 4,'não está relacionado' : 5},)

        df_6 = df_5.copy()
        for c in df_6:
            df_6[c] = pd.to_numeric(df_6[c], errors = 'coerce')

        df_7 = df_4[["Nome"]].copy()

        #constrói as respectivas colunas a patir da média das colunas que foram listadas como pertencentes a cada uma das quatro dimensões
        for c in [c_Acuracia, c_Cobertura, c_Precisao, c_Relevância]:
            df_7[c[0].split()[0]] = df_6[list(set(c))].mean(axis=1)

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
        df_11["Conceito"] = arquivo.split("ontologia ")[1].split("_1")[0]
        df_11.set_index("Conceito", inplace = True)
        df_12 = pd.concat([df_12, df_11])
    except:
        print('Falha')