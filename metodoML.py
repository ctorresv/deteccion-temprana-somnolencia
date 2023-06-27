# -*- coding: utf-8 -*-

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

bos = 0.6
per= 0.19
in1 = 1
inB = 1


def fuzyLogic(per,bos,in1,inB):

    # Generar variables del universo de discurso
    x_bostezo = np.arange(0,1, 0.1)
    x_perclos = np.arange(0,1, 0.1)
    x_indi1 = np.arange(0,1,0.1)
    x_indB = np.arange(0,1,0.1)
    x_class  = np.arange(0, 1, 0.1)
    
    # Generar funciones de membresía difusas
    # ---------- boca ----------------------
    bocaNorm = fuzz.smf(x_bostezo, 0, 0.2)
    bocaHabl = fuzz.smf(x_bostezo, 0.3, 0.49)
    bocaBos = fuzz.smf(x_bostezo, 0.5, 1)
    
    # ---------- ojos ----------------------
    #perclosCansado = fuzz.zmf(x_perclos, 0,0.80)
    #perclosNormal = fuzz.smf(x_perclos, 0.90, 1)
    
    perclosCansado = fuzz.trapmf(x_perclos, [0.0, 0.40, 0.60, 0.80])
    perclosNormal = fuzz.trapmf(x_perclos, [0.60, 0.80, 0.90, 1])
    
    # ---------- indicadorOjos ------------------------
    #indiOPEN = fuzz.zmf(x_indi1,  0,0.3)
    #indiCLOSE = fuzz.zmf(x_indi1, 0.4,1)
    
    # ---------- indicadorBostezo ------------------------
    
    #indiBostezoNO = fuzz.zmf(x_indB, 0,0.3)
    #indiBostezoSI = fuzz.zmf(x_indB, 0.4,1)
    
    indiOPEN = fuzz.trapmf(x_indi1, [0,0,0.2,0.3])
    indiCLOSE = fuzz.trapmf(x_indi1, [0.2,0.3,1,1])
    
    # ---------- indicadorBostezo ------------------------
    
    indiBostezoNO = fuzz.trapmf(x_indB, [0,0,0.2,0.3])
    indiBostezoSI = fuzz.trapmf(x_indB, [0.2,0.3,1,1])
    
    
    # ---------- class ------------------------
    class_normal = fuzz.zmf(x_class, 0, 0.5)
    class_sueño = fuzz.smf(x_class, 0.6, 1)
    '''
        # Visualizando estos universos y funciones de membresía
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 9))
    ax0.plot(x_bostezo, bocaNorm, 'y', linewidth=1.5, label='Normar')
    ax0.plot(x_bostezo, bocaHabl, 'b', linewidth=1.5, label='Habla')
    ax0.plot(x_bostezo, bocaBos, 'k', linewidth=1.5, label='Bostezo')
    ax0.set_title('Bostezo')
    ax0.legend()
    
    ax1.plot(x_perclos, perclosNormal, 'y', linewidth=1.5, label='Normal')
    ax1.plot(x_perclos, perclosCansado, 'b', linewidth=1.5, label='Cerrado')
    ax1.set_title('perCLOS')
    ax1.legend()
    
    ax2.plot(x_indi1, indiOPEN, 'y', linewidth=1.5, label='Abierto')
    ax2.plot(x_indi1, indiCLOSE, 'b', linewidth=1.5, label='Cerrado')
    ax2.set_title('Indicador de OJO')
    ax2.legend()
    
    ax3.plot(x_indB, indiBostezoNO, 'y', linewidth=1.5, label='No Bostezo')
    ax3.plot(x_indB, indiBostezoSI, 'b', linewidth=1.5, label='Si Bostezo')
    ax3.set_title('Indicador de Bostezo')
    ax3.legend()
    
    
    ax4.plot(x_class, class_normal, 'y', linewidth=1.5, label='Normal')
    ax4.plot(x_class, class_sueño, 'b', linewidth=1.5, label='Sueño')
    ax4.set_title('clase de salida')
    ax4.legend()
    
    # Quitar ejes
    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    plt.tight_layout()
    '''
      
    # Definimos antecedentes y consecuentes
    
    perClos = ctrl.Antecedent(np.arange(0,1, 0.1),'perCLOS')
    Bostezo = ctrl.Antecedent(np.arange(0,1, 0.1), 'Bostezo')
    indicadorOJOS  = ctrl.Antecedent(np.arange(0,1,0.1), 'Indi1')
    indicadorBostezo = ctrl.Antecedent(np.arange(0,1,0.1), 'IndB')
    Class  = ctrl.Consequent(np.arange(0, 1, 0.1), 'CLASS')
    
    # Definimos las variables de antecedentes y consecuentes
    perClos['Normal'] = perclosNormal
    perClos['Cansado'] = perclosCansado
    indicadorOJOS['Abierto'] = indiOPEN
    indicadorOJOS['Cerrados'] = indiCLOSE
    indicadorBostezo['NoBostezo'] = indiBostezoNO
    indicadorBostezo['SiBostezo'] = indiBostezoSI
    Bostezo['Normal'] = bocaNorm
    Bostezo['Habla'] = bocaHabl
    Bostezo['Bostezo'] = bocaBos
    Class['Normal'] = class_normal
    Class['Sueño'] = class_sueño

    # Establecemos las reglas
    
    rule1 = ctrl.Rule((perClos['Normal'] & indicadorOJOS['Abierto']) | (Bostezo['Normal'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule2 = ctrl.Rule(perClos['Normal'] & Bostezo['Normal'], Class['Normal'])
    rule3 = ctrl.Rule(perClos['Cansado'] & indicadorOJOS['Cerrados'], Class['Sueño'])
    rule4 = ctrl.Rule((perClos['Cansado'] & Bostezo['Normal']) | (indicadorOJOS['Cerrados'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule5 = ctrl.Rule(perClos['Cansado'] & Bostezo['Habla'] & indicadorOJOS['Cerrados'] & indicadorBostezo['NoBostezo'], Class['Sueño'])
    rule6 = ctrl.Rule(perClos['Cansado'] & Bostezo['Bostezo'], Class['Sueño'])
    rule7 = ctrl.Rule(perClos['Cansado'] & Bostezo['Normal'] & indicadorOJOS['Cerrados'], Class['Sueño'])
    
    '''
    
    
    rule1 = ctrl.Rule((perClos['Normal'] & indicadorOJOS['Abierto']) | (Bostezo['Normal'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule2 = ctrl.Rule((perClos['Normal'] & indicadorOJOS['Abierto']) | (Bostezo['Habla'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule3 = ctrl.Rule((perClos['Normal'] & indicadorOJOS['Abierto']) | (Bostezo['Bostezo'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule4 = ctrl.Rule((perClos['Normal'] & indicadorOJOS['Abierto']) | (Bostezo['Bostezo'] & indicadorBostezo['SiBostezo']), Class['Sueño'])
    rule5 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Abierto']) | (Bostezo['Normal'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule6 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Abierto']) | (Bostezo['Habla'] & indicadorBostezo['NoBostezo']), Class['Normal'])
    rule7 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Abierto']) | (Bostezo['Bostezo'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule8 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Cerrados']) | (Bostezo['Normal'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule9 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Cerrados']) | (Bostezo['Habla'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule10 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Cerrados']) | (Bostezo['Bostezo'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule11 = ctrl.Rule((perClos['Cansado'] & indicadorOJOS['Cerrados']) | (Bostezo['Bostezo'] & indicadorBostezo['SiBostezo']), Class['Normal'])
    rule12 = ctrl.Rule(perClos['Normal'] & Bostezo['Normal'], Class['Normal'])
    rule13 = ctrl.Rule(perClos['Normal'] & indicadorOJOS['Abierto'], Class['Normal'])
    rule14 = ctrl.Rule(perClos['Normal'] & Bostezo['Habla'], Class['Normal'])
    rule15 = ctrl.Rule(indicadorOJOS['Abierto'] & indicadorBostezo['NoBostezo'], Class['Normal'])
    rule16 = ctrl.Rule((perClos['Cansado'] & Bostezo['Normal']) | (indicadorOJOS['Cerrados'] & indicadorBostezo['NoBostezo']), Class['Sueño'])
    rule17 = ctrl.Rule(perClos['Cansado'] & Bostezo['Habla'] & indicadorOJOS['Cerrados'] & indicadorBostezo['NoBostezo'], Class['Sueño'])
    rule18 = ctrl.Rule(perClos['Cansado'] & Bostezo['Bostezo'], Class['Sueño'])
    rule19 = ctrl.Rule(indicadorOJOS['Cerrados'] & indicadorBostezo['SiBostezo'], Class['Sueño'])
    rule20 = ctrl.Rule(perClos['Cansado'] & indicadorOJOS['Cerrados'], Class['Normal'])
    '''
    Sueño_ctrl = ctrl.ControlSystem([rule1,rule2,rule4,rule5,rule6,rule7])
    #Sueño_ctrl = ctrl.ControlSystem([rule1,rule2,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11])
    #Sueño_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11
    #,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20])
    # Creamos el Simulador del "sistema de control"
    sueño = ctrl.ControlSystemSimulation(Sueño_ctrl)
    sueño.input['perCLOS'] = per
    sueño.input['Bostezo'] = bos
    sueño.input['Indi1'] = in1
    sueño.input['IndB'] = inB
        
    sueño.compute()
    #print(per,bos,in1,inB)
    print("Clase: ",sueño.output['CLASS'])
    return sueño.output['CLASS']
fuzyLogic(per,bos,in1,inB)