import re
import pandas as pd
import spacy
import skweak
from spacy.tokens import DocBin

# Classe com todas Label funcitons para extracao de contratos


class LabelFunctionsContratos:
    def __init__(self, arquivo):
        nlp = spacy.load('pt_core_news_sm', disable=["ner", "lemmatizer"])
        self.docs = list(nlp.pipe(arquivo))

    # Label functions Vitor Araruna
    def contrato_(self, doc):
        expression = r"(, doc\d+/\d{4})"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                tamanho = len(grupo.split())
                for token in doc:
                    if token.text in {grupo}:
                        yield token.i-1, token.i+tamanho, "CONTRATO"
                        # print(docs[0][token.i:token.i+tamanho])
                break

    def processo_(self, doc):
        expression = r"[P|p][R|r][O|o][C|c][E|e][S|s][S|s][O|o][:|\sSEI].*?(\d+[\.|-|–|\/]\d+[\.|-|–|\/]\d+[\.|-|–|\/]\d*)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'\/\s', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "PROCESSO"
                        break
                break

    def data_assinatura_(self, doc):
        expression = r"[A|a][S|s][S|s][I|i][N|n][A|a][T|t][U|u][R|r][A|a]:.*?[\s\S](\d{2}\/\d{2}\/\d{4}|\d+.*?[\s\S]\w+.*?[\s\S]\w+.*?[\s\S]\w+.*?[\s\S]\d+)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'\/\s', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "DATA_ASS."
                        break
                break

    def vigencia_(self, doc):
        expression = r"[V|v][I|i][G|g][E|e|ê][N|n][C|c][I|i][A|a].*?[\S\s].*?([^,|;|.]*)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "VIGENCIA"
                        break
                break

    def valor_(self, doc):
        expression = r"[v|V][a|A][l|L][o|O][r|R].*?[\s\S].*?([\d\.]*,\d{2})"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i-1, token.i+tamanho+len(x), "VALOR"
                        break
                break

    def unidade_orcamento_(self, doc):
        expression = r"[o|O][r|R][c|C|ç|Ç][a|A][m|M][e|E][n|N][t|T][a|A|á|Á][r|R][i|I][a|A].*?[\s\S].*?(\d+.\d+)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "UNI_ORC."
                        break
                break

    def programa_trabalho_(self, doc):
        expression = r"[T|t][R|r][A|a][B|b][A|a][L|l][H|h][O|o]?[:|;|[\s\S].*?(\d*\.\d*.\d*.\d*.\d{4,6}.\d{4,6})"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "PROG_TRAB."
                        break
                break

    def natureza_emepnho_(self, doc):
        expression = r"Natureza d[e|a] Despesa:[\s\S].*?([\d.\d]*)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "NAT_DESP."
                        break
                break

    def nota_empenho_(self, doc):
        expression = r"(\d+NE\d+)"
        # doc.tex with m re.finditer to return span with char_span
        for match in re.finditer(expression, doc.text):
            if(match.groups):
                grupo = (match.groups()[0])
                grupo_copy = grupo
                if ("(" or ")") in grupo:
                    grupo = grupo.replace("(", "\(")
                    grupo = grupo.replace(")", "\)")
                tamanho = len(grupo.split())
                start = re.search(grupo, doc.text).span()[0]
                end = re.search(grupo, doc.text).span()[1]
                span = str(doc.char_span(start, end))
                x = re.findall(r'[\(|\)]', span)
                for token in doc:
                    if(grupo_copy in str(doc[token.i: token.i+tamanho+len(x)])):
                        yield token.i, token.i+tamanho+len(x), "NOTA_EMP."
                        break
                break

    # Label functions Vitor Vasconcelos
    def contrato_detector_fun(self, doc):
        for token in doc:
            if token.text in {'CONTRATO'}:
                for x in range((len(doc)-token.i-1)):
                    if doc[token.i+x].text in {'No', 'NO', 'no', 'Nº', 'nº', 'N°', 'n°'} and doc[token.i+x+1].text[0].isdigit():
                        yield doc[token.i+x].i, doc[token.i+x].i+2, "CONTRATO",
                        break

    def processo_detector_fun(self, doc):
        for token in doc:
            for y in ['Processo', 'PROCESSO', 'Processo:', 'PROCESSO:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('} or doc[token.i+x].text in {'Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK'}:
                            yield token.i+1+k, token.i+x, "PROCESSO"
                            break

    def partes_detector_fun(self, doc):
        for token in doc:
            for y in ['Partes', 'PARTES', 'partes:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', ';'} or doc[token.i+x].text in {'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', '<>END OF BLOCK<>', 'END OF BLOCK'}:
                            yield token.i+1+k, token.i+x, "PARTES"
                            break

    def objeto_detector_fun(self, doc):
        flag = 0
        for token in doc:
            if token.text in {'Objeto', 'OBJETO'}:
                k = 0
                if(doc[token.i+1].text == ':'):
                    k = 1
                for x in range((len(doc)-token.i-1)):
                    if doc[token.i+x+1].text in {'Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK'}:  # {'.'}:
                        flag = 1
                        yield token.i+1+k, token.i+x-1, "OBJETO"
                        break
            if token.text in {'tem', 'Tem'} and doc[token.i+1].text in {'por', 'POR'} and doc[token.i+2].text in {'objeto', 'Objeto', 'OBJETO'} and flag == 0:
                for x in range((len(doc)-token.i-1)):
                    if doc[token.i+x+1].text in {'Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK'}:  # {'.'}:
                        yield token.i, token.i+x, "OBJETO"
                        break

    def valor_detector_fun(self, doc):
        for token in doc:
            for y in ['Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:']:
                if y in token.text:
                    for x in range((len(doc)-token.i-1)):
                        if doc[token.i+x].text in {'R$', '$', '$$'} and doc[token.i+x+1].text[0].isdigit():
                            yield doc[token.i+x].i, doc[token.i+x].i+2, "VALOR",
                            break

    def lei_orc_detector_fun(self, doc):
        for token in doc:
            for y in ['Lei', 'LEI', 'lei:', 'Lei:', 'LEI:']:
                if y in token.text and doc[token.i+1].text in {'Orçamentária', 'ORÇAMENTÁRIA', 'ORÇAMENTARIA', 'oçamentária', 'Orçamentária:', 'ORÇAMENTÁRIA:', 'ORÇAMENTARIA:', 'oçamentária:'}:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+2+k, token.i+x, "LEI_ORC."
                            break

    def dotacao_orc_detector_fun(self, doc):
        for token in doc:
            for y in ['Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao']:
                if y in token.text and doc[token.i+1].text in {'Orçamentária', 'ORÇAMENTÁRIA', 'ORÇAMENTARIA', 'oçamentária', 'Orçamentária:', 'ORÇAMENTÁRIA:', 'ORÇAMENTARIA:', 'oçamentária:'}:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+3, token.i+x, "DOT_ORC."
                            break

    def unidade_orc_detector_fun(self, doc):
        for token in doc:
            for y in ['Unidade', 'UNIDADE', 'unidade']:
                if y in token.text and doc[token.i+1].text in {'Orçamentária', 'Orcamentaria', 'ORÇAMENTÁRIA', 'ORCAMENTARIA', 'orcamentaria', 'orçamentária', 'Orçamentária:', 'Orcamentaria:', 'ORÇAMENTÁRIA:', 'ORCAMENTARIA:', 'orcamentaria:', 'orçamentária:'}:
                    k = 0
                    if(doc[token.i+2].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+2+k, token.i+x, "UNI_ORC."
                            break

    def programa_trab_detector_fun(self, doc):
        for token in doc:
            for y in ['Programa', 'PROGRAMA', 'programa']:
                if y in token.text and doc[token.i+1].text in {'de', 'do', 'da', 'DE', 'DO', 'DA'} and doc[token.i+2].text in {'trabalho', 'Trabalho', 'TRABALHO', 'trabalho:', 'Trabalho:', 'TRABALHO:'}:
                    k = 0
                    if(doc[token.i+3].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+3+k, token.i+x, "PROG_TRAB."
                            break

    def natureza_desp_detector_fun(self, doc):
        for token in doc:
            for y in ['Natureza', 'NATUREZA', 'natureza']:
                if y in token.text and doc[token.i+1].text in {'de', 'do', 'da', 'DE', 'DO', 'DA', 'DAS', 'das'} and doc[token.i+2].text in {'despesa', 'Despesa', 'DESPESA', 'despesa:', 'Despesa:', 'DESPESA:', 'despesas', 'Despesas', 'DESPESAs', 'despesas:', 'Despesas:', 'DESPESAs:'}:
                    k = 0
                    if(doc[token.i+3].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+3+k, token.i+x, "NAT_DESP."
                            break

    def data_detector_fun(self, doc):
        for token in doc:
            for y in ['Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '(', 'Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK'}:
                            yield token.i+1+k, token.i+x, "DATA_ASS."
                            break

    def signatarios_detector_fun(self, doc):
        for token in doc:
            for y in ['SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i-2)):
                        if '.' in doc[token.i+x+2].text:
                            yield token.i+1+k, token.i+x+3, "SIGNATARIOS"
                            break

    def vigencia_detector_fun(self, doc):
        for token in doc:
            # 'Vigencia','Vigência'
            for y in ['VIGÊNCIA', 'VIGENCIA', 'Vigência:', 'VIGÊNCIA:', 'VIGENCIA:', 'Vigencia:', 'Vigencia']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ';'}:
                            yield token.i+1+k, token.i+x, "VIGENCIA"
                            break

    def nota_emp_detector_fun(self, doc):
        for token in doc:
            for y in ['Empenho', 'EMPENHO', 'Empenho:', 'EMPENHO:', 'empenho:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+1+k, token.i+x, "NOTA_EMP."
                            break

    def numero_ajuste_detector_fun(self, doc):
        for token in doc:
            for y in ['número', 'numero', 'Número', 'Numero', 'NÚMERO', 'NUMERO']:
                if y in token.text and doc[token.i+1].text in {'de', 'do', 'da', 'DE', 'DO', 'DA'} and doc[token.i+2].text in {'Ajuste', 'AJUSTE', 'ajuste', 'Ajuste:', 'AJUSTE:', 'ajuste:'}:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'.', '-', '–', ',', ';', '('}:
                            yield token.i+3+k, token.i+x, "NUM_AJUSTE."
                            break

    def procedimento_detector_fun(self, doc):
        for token in doc:
            for y in ['Procedimento', 'PROCEDIMENTO', 'PROCEDIMENTO:', 'procedimento:', 'Procedimento:']:
                if y in token.text:
                    k = 0
                    if(doc[token.i+1].text == ':'):
                        k = 1
                    for x in range((len(doc)-token.i)):
                        if doc[token.i+x].text in {'Partes', 'PARTES', 'partes:', 'Objeto', 'OBJETO', 'Valor', 'VALOR', 'Valor:', 'VALOR:', 'valor:', 'Lei', 'LEI', 'lei:', 'Lei:', 'LEI:', 'Dotação', 'DOTAÇÃO', 'dotação', 'DOTACAO', 'Dotacao', 'dotacao', 'Unidade', 'UNIDADE', 'unidade', 'Programa', 'PROGRAMA', 'programa', 'Natureza', 'NATUREZA', 'natureza', 'Data', 'DATA', 'data', 'Assinatura', 'ASSINATURA', 'assinatura:', 'Assinatura:', 'ASSINATURA:', 'SIGNATÁRIOS', 'SIGNATARIOS', 'Signatários:', 'SIGNATÁRIOS:', 'SIGNATARIOS:', 'Signatarios:', 'Signatarios', 'Assinantes', 'ASSINANTES', 'Assinantes:', 'ASSINANTES:', '<>END OF BLOCK<>', 'END OF BLOCK'}:
                            yield token.i+1+k, token.i+x, "PROCED."
                            break

    def money_detector_fun(self, doc):
        for token in doc:
            if token.text[0].isdigit() and doc[token.i-1].text in {'R$', '$', '$$'}:
                yield token.i-1, token.i+1, "DINHEIRO"

# Classe que aplica o modelo
# Para seu funcionamento idela, eh necessario inicializa-da com um dataset de contratos


class SkweakContratos(LabelFunctionsContratos):
    def __init__(self, arquivo):
        # Inicializa o docs e o dataframe
        super().__init__(arquivo)
        self.df = pd.DataFrame(columns=["CONTRATO", "PROCESSO", "PARTES", "OBJETO", "VALOR", "LEI_ORC.", "DOT_ORC.", "UNI_ORC.",
                               "PROG_TRAB.", "NAT_DESP.", "DATA_ASS.", "SIGNATARIOS", "VIGENCIA", "NOTA_EMP.", "NUM_AJUSTE.", "PROCED."])

    # Aplica as label functions
    def apply_label_functions(self):
        # Aplicando cada uma no dataset
        doc = self.docs
        detec_contrato = skweak.heuristics.FunctionAnnotator(
            "detec_contrato", self.contrato_)
        doc = list(detec_contrato.pipe(doc))

        detec_processo = skweak.heuristics.FunctionAnnotator(
            "detec_processo", self.processo_)
        doc = list(detec_processo.pipe(doc))

        detec_data = skweak.heuristics.FunctionAnnotator(
            "detec_data", self.data_assinatura_)
        doc = list(detec_data.pipe(doc))

        detec_vigencia = skweak.heuristics.FunctionAnnotator(
            "detec_vigencia", self.vigencia_)
        doc = list(detec_vigencia.pipe(doc))

        detec_valor = skweak.heuristics.FunctionAnnotator(
            "detec_valor", self.valor_)
        doc = list(detec_valor.pipe(doc))

        detec_unidade = skweak.heuristics.FunctionAnnotator(
            "detec_unidade", self.unidade_orcamento_)
        doc = list(detec_unidade.pipe(doc))

        detec_programa = skweak.heuristics.FunctionAnnotator(
            "detec_programa", self.programa_trabalho_)
        doc = list(detec_programa.pipe(doc))

        detec_natureza = skweak.heuristics.FunctionAnnotator(
            "detec_natureza", self.natureza_emepnho_)
        doc = list(detec_natureza.pipe(doc))

        detec_nota = skweak.heuristics.FunctionAnnotator(
            "detec_nota", self.nota_empenho_)
        doc = list(detec_nota.pipe(doc))

        # Aplicando cada uma no dataset
        contrato_detector = skweak.heuristics.FunctionAnnotator(
            "contrato_detector", self.contrato_detector_fun)
        doc = list(contrato_detector.pipe(doc))

        processo_detector = skweak.heuristics.FunctionAnnotator(
            "processo_detector", self.processo_detector_fun)
        doc = list(processo_detector.pipe(doc))

        partes_detector = skweak.heuristics.FunctionAnnotator(
            "partes_detector", self.partes_detector_fun)
        doc = list(partes_detector.pipe(doc))

        objeto_detector = skweak.heuristics.FunctionAnnotator(
            "objeto_detector", self.objeto_detector_fun)
        doc = list(objeto_detector.pipe(doc))

        valor_detector = skweak.heuristics.FunctionAnnotator(
            "valor_detector", self.valor_detector_fun)
        doc = list(valor_detector.pipe(doc))

        lei_orc_detector = skweak.heuristics.FunctionAnnotator(
            "lei_orc_detector", self.lei_orc_detector_fun)
        doc = list(lei_orc_detector.pipe(doc))

        unidade_orc_detector = skweak.heuristics.FunctionAnnotator(
            "unidade_orc_detector", self.unidade_orc_detector_fun)
        doc = list(unidade_orc_detector.pipe(doc))

        programa_trab_detector = skweak.heuristics.FunctionAnnotator(
            "programa_trab_detector", self.programa_trab_detector_fun)
        doc = list(programa_trab_detector.pipe(doc))

        natureza_desp_detector = skweak.heuristics.FunctionAnnotator(
            "natureza_desp_detector", self.natureza_desp_detector_fun)
        doc = list(natureza_desp_detector.pipe(doc))

        data_detector = skweak.heuristics.FunctionAnnotator(
            "data_detector", self.data_detector_fun)
        doc = list(data_detector.pipe(doc))

        signatarios_detector = skweak.heuristics.FunctionAnnotator(
            "signatarios_detector", self.signatarios_detector_fun)
        doc = list(signatarios_detector.pipe(doc))

        vigencia_detector = skweak.heuristics.FunctionAnnotator(
            "vigencia_detector", self.vigencia_detector_fun)
        doc = list(vigencia_detector.pipe(doc))

        nota_emp_detector = skweak.heuristics.FunctionAnnotator(
            "nota_emp_detector", self.nota_emp_detector_fun)
        doc = list(nota_emp_detector.pipe(doc))

        numero_ajuste_detector = skweak.heuristics.FunctionAnnotator(
            "numero_ajuste_detector", self.numero_ajuste_detector_fun)
        doc = list(numero_ajuste_detector.pipe(doc))

        procedimento_detector = skweak.heuristics.FunctionAnnotator(
            "procedimento_detector", self.procedimento_detector_fun)
        doc = list(procedimento_detector.pipe(doc))

        money_detector = skweak.heuristics.FunctionAnnotator(
            "money_detector", self.money_detector_fun)
        doc = list(money_detector.pipe(doc))

        self.docs = doc

    #  Treina o modelo em HMM
    def train_HMM_Dodf(self):
        # Calculo do algoritimo HMM e agregacao das funcoes
        model = skweak.aggregation.HMM("hmm", ["CONTRATO", "PROCESSO", "PARTES", "OBJETO", "VALOR", "LEI ORC.", "DOT ORC.", "UNI ORC.", "PROG TRAB.",
                                               "NAT DESP.", "DATA ASS.", "SIGNATARIOS", "VIGENCIA", "NOTA EMP.", "NUM AJUSTE.", "PROCED."], sequence_labelling=True)  # ,"DINHEIRO"

        self.docs = model.fit_and_aggregate(self.docs)

        # Salvando modelo HMM para treinamento
        for doc in self.docs:
            doc.ents = doc.spans["hmm"]
        skweak.utils.docbin_writer(self.docs, "./data/reuters_small.spacy")

    # retorna os resultados das entidades em IOB
    def get_IOB(self):
        # Mostrando entidades no modelo em IOB
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/reuters_small.spacy")
        examples = []

        for doc in doc_bin.get_docs(nlp.vocab):
            lista_iob = []
            for i in range(0, len(doc)):
                label_iob = ""
                _txt_ = doc[i].text
                _label_ = doc[i].ent_iob_
                _ent_ = doc[i].ent_type_
                if _txt_ not in ["", " ", "  ", "   "]:
                    if(_label_ != "O"):
                        label_iob += f'{_txt_} {_label_}-{_ent_}'
                    else:
                        label_iob += f'{_txt_} {_label_}'
                    lista_iob.append(label_iob)
            examples.append(lista_iob)

        return examples

    # Mostra os spans hmm de um doc especifico
    def list_spans_specific(self, x):
        print(self.docs[x].spans["hmm"])

    # Mostra os spans hmm do doc inteiro
    def list_spans_all(self):
        for doc in self.docs:
            print(doc["hmm"])

    # Retorna o dataframe com todas as entidades, textos e IOB para cada documento
    def get_hmm_dataframe(self):
        nlp = spacy.blank("pt")
        doc_bin = DocBin().from_disk("./data/reuters_small.spacy")

        for doc in doc_bin.get_docs(nlp.vocab):
            aux = {"CONTRATO": "", "PROCESSO": "", "PARTES": "", "OBJETO": "", "VALOR": "", "LEI_ORC.": "", "DOT_ORC.": "", "UNI_ORC.": "", "PROG_TRAB.": "",
                   "NAT_DESP.": "", "DATA_ASS.": "", "SIGNATARIOS": "", "VIGENCIA": "", "NOTA_EMP.": "", "NUM_AJUSTE.": "", "PROCED.": "", "text": "", "labels": ""}

            for entity in doc.ents:
                aux[entity[0].ent_type_] = entity.text

            for token in doc:
                aux["text"] += token.text + ' '

            for i in range(0, len(doc)):
                _txt_ = doc[i].text
                _label_ = doc[i].ent_iob_
                _ent_ = doc[i].ent_type_
                if _txt_ not in ["", " ", "  ", "   "]:
                    if(_label_ != "O"):
                        aux["labels"] += f'{_label_}-{_ent_} '
                    else:
                        aux["labels"] += f'{_label_} '

            self.df = self.df.append(aux, ignore_index=True)

        return self.df

    # Salva o dataframe em um .csv
    def save_dataframe_csv(self):
        self.df.to_csv('ContratosHMM.csv')
