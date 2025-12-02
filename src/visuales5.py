"""
Visualización Avanzada de Difusión y Contagio de Crisis en Mercado de Opciones
Enfoque físico-matemático con visualizaciones profesionales
"""

import solara
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import networkx as nx
from scipy import signal
from scipy.stats import gaussian_kde
import threading
import time

# set the route to import scripts to the parent directory
import style
style.set_style()



# Importar el modelo
from options_market_abm import OptionsMarketModel


# Variables reactivas
running = solara.reactive(False)
step_counter = solara.reactive(0)

# Parámetros del modelo
n_market_makers = solara.reactive(7)
n_speculators = solara.reactive(40)
n_hedgers = solara.reactive(15)
n_leveraged = solara.reactive(25)
initial_underlying_price = solara.reactive(100.0)
initial_volatility = solara.reactive(0.20)
shock_step = solara.reactive(50)
shock_magnitude = solara.reactive(-0.10)
seed = solara.reactive(42)

# Modelo y datos
model = solara.reactive(None)
model_data = solara.reactive(pd.DataFrame())
agent_states_history = solara.reactive([])

simulation_thread = None


def get_agent_state(agent):
    """Obtener estado del agente para visualización"""
    from options_market_abm import MarketMaker, Speculator, Hedger, Leveraged
    
    if isinstance(agent, MarketMaker):
        health = agent.capital / agent.initial_capital if agent.initial_capital > 0 else 0
        return {
            'type': 'Market Maker',
            'health': health,
            'stress': abs(agent.inventory) / agent.max_inventory if agent.max_inventory > 0 else 0,
            'liquidated': False,
            'capital': agent.capital,
            'inventory': agent.inventory
        }
    elif isinstance(agent, Speculator):
        health = agent.capital / agent.initial_capital if agent.initial_capital > 0 else 0
        return {
            'type': 'Speculator',
            'health': health,
            'stress': abs(agent.position) / agent.max_position if agent.max_position > 0 else 0,
            'liquidated': False,
            'capital': agent.capital,
            'position': agent.position
        }
    elif isinstance(agent, Hedger):
        return {
            'type': 'Hedger',
            'health': 1.0,
            'stress': agent.hedge_ratio,
            'liquidated': False,
            'capital': agent.stock_portfolio_value,
            'position': agent.options_position
        }
    elif isinstance(agent, Leveraged):
        health = agent.own_capital / agent.initial_capital if agent.initial_capital > 0 else 0
        return {
            'type': 'Leveraged',
            'health': health,
            'stress': abs(agent.position) * 0.01 if not agent.liquidated else 1.0,
            'liquidated': agent.liquidated,
            'capital': agent.own_capital,
            'position': agent.position
        }
    return {'type': 'Unknown', 'health': 1.0, 'stress': 0, 'liquidated': False, 'capital': 0, 'position': 0}


def initialize_model():
    """Inicializar el modelo"""
    new_model = OptionsMarketModel(
        n_market_makers=n_market_makers.value,
        n_speculators=n_speculators.value,
        n_hedgers=n_hedgers.value,
        n_leveraged=n_leveraged.value,
        initial_underlying_price=initial_underlying_price.value,
        initial_volatility=initial_volatility.value,
        shock_step=shock_step.value,
        shock_magnitude=shock_magnitude.value,
        seed=seed.value
    )
    model.value = new_model
    model_data.value = pd.DataFrame()
    agent_states_history.value = []
    step_counter.value = 0


def step_model():
    """Ejecutar un paso del modelo"""
    if model.value is not None:
        try:
            model.value.step()
            step_counter.value += 1
            
            # Recolectar datos
            current_data = model.value.datacollector.get_model_vars_dataframe()
            model_data.value = current_data
            
            # Guardar estados de agentes
            agent_states = [get_agent_state(agent) for agent in model.value.agents]
            history = agent_states_history.value.copy()
            history.append({
                'step': step_counter.value,
                'states': agent_states,
                'in_crisis': model.value.in_crisis,
                'margin_calls': model.value.margin_calls,
                'price': model.value.underlying_price,
                'volatility': model.value.current_volatility
            })
            agent_states_history.value = history
            
        except Exception as e:
            print(f"Error en step_model: {e}")


def reset_model():
    """Resetear el modelo"""
    global simulation_thread
    running.value = False
    if simulation_thread and simulation_thread.is_alive():
        simulation_thread.join(timeout=1.0)
    initialize_model()


def play_model():
    """Iniciar/pausar la simulación"""
    global simulation_thread
    
    if running.value:
        running.value = False
        if simulation_thread and simulation_thread.is_alive():
            simulation_thread.join(timeout=1.0)
    else:
        running.value = True
        simulation_thread = threading.Thread(target=run_simulation, daemon=True)
        simulation_thread.start()


def run_simulation():
    """Ejecutar simulación en thread separado"""
    while running.value:
        step_model()
        time.sleep(0.12)


@solara.component
def DiffusionVisualizationApp():
    """Aplicación principal de visualización de difusión"""
    
    if model.value is None:
        initialize_model()
    
    with solara.Column(style={"width": "100%", "background": "#ffffff"}):
        
        # Header profesional
        with solara.Card(style={"background": "#1e3a8a", "color": "white", "padding": "20px", "margin-bottom": "20px"}):
            solara.Markdown("# Crisis Diffusion Dynamics in Options Markets")
            solara.Markdown("*Agent-Based Model with Geometric Brownian Motion Process*")
        
        with solara.Columns([1, 4]):
            
            # Panel de control
            with solara.Column(style={"padding": "20px", "background": "#f8fafc"}):
                
                solara.Markdown("### Control Panel")
                with solara.Column(gap="10px"):
                    solara.Button("RESET", on_click=reset_model, color="primary", style={"width": "100%"})
                    solara.Button(
                        "PAUSE" if running.value else "RUN",
                        on_click=play_model,
                        color="warning" if running.value else "success",
                        style={"width": "100%"}
                    )
                    solara.Button("STEP", on_click=step_model, color="primary", 
                                disabled=running.value, style={"width": "100%"})
                
                solara.Markdown("---")
                solara.Markdown(f"**Current Step:** {step_counter.value}")
                
                if model.value is not None:
                    crisis_status = "CRISIS" if model.value.in_crisis else "Normal"
                    status_color = "#ef4444" if model.value.in_crisis else "#10b981"
                    
                    with solara.Card(style={"background": "#ffffff", "border": f"2px solid {status_color}"}):
                        solara.Markdown(f"**Market Status:** {crisis_status}")
                        solara.Markdown(f"**Underlying Price:** ${model.value.underlying_price:.2f}")
                        solara.Markdown(f"**Implied Volatility:** {model.value.current_volatility*100:.2f}%")
                        solara.Markdown(f"**Margin Calls:** {model.value.margin_calls}")
                
                solara.Markdown("---")
                solara.Markdown("### Model Parameters")
                
                with solara.Column(gap="8px"):
                    solara.SliderInt("Market Makers", value=n_market_makers, min=3, max=15, disabled=running.value)
                    solara.SliderInt("Speculators", value=n_speculators, min=10, max=80, disabled=running.value)
                    solara.SliderInt("Hedgers", value=n_hedgers, min=5, max=40, disabled=running.value)
                    solara.SliderInt("Leveraged Traders", value=n_leveraged, min=10, max=80, disabled=running.value)
                    solara.SliderInt("Shock Step", value=shock_step, min=20, max=100, disabled=running.value)
                    solara.SliderFloat("Shock Magnitude", value=shock_magnitude, min=-0.30, max=0.30, step=0.05, disabled=running.value)
            
            # Visualizaciones
            with solara.Column():
                
                if len(agent_states_history.value) > 0:
                    
                    # Fila 1: Red + Phase Space
                    with solara.Columns([1, 1]):
                        PlotAgentNetwork()
                        PlotPhaseSpace()
                    
                    # Fila 2: Cascadas + Distribuciones
                    with solara.Columns([1, 1]):
                        PlotLiquidationCascades()
                        PlotAgentDistributions()
                    
                    # Fila 3: Análisis Espectral + Contagio
                    with solara.Columns([1, 1]):
                        PlotSpectralAnalysis()
                        PlotContagionMetrics()
                    
                else:
                    with solara.Card():
                        solara.Info("Click RUN or STEP to begin simulation and observe crisis propagation dynamics")


@solara.component
def PlotAgentNetwork():
    """Red de agentes con visualización mejorada"""
    history = agent_states_history.value
    
    if len(history) == 0:
        return
    
    current_states = history[-1]['states']
    
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    # Crear grafo
    G = nx.Graph()
    
    # Agrupar agentes por tipo
    type_groups = {}
    for i, state in enumerate(current_states):
        agent_type = state['type']
        if agent_type not in type_groups:
            type_groups[agent_type] = []
        type_groups[agent_type].append(i)
    
    # Añadir nodos
    for i in range(len(current_states)):
        G.add_node(i)
    
    # Crear conexiones basadas en similitud de estrés
    for i in range(len(current_states)):
        for j in range(i+1, len(current_states)):
            stress_diff = abs(current_states[i]['stress'] - current_states[j]['stress'])
            if stress_diff < 0.3:  # Conectar agentes con niveles de estrés similares
                G.add_edge(i, j, weight=1-stress_diff)
    
    # Layout spring
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Colores según salud
    node_colors = []
    node_sizes = []
    edge_widths = []
    
    for state in current_states:
        if state['liquidated']:
            node_colors.append('#1f2937')
            node_sizes.append(100)
        else:
            health = state['health']
            if health > 0.7:
                node_colors.append('#10b981')
                node_sizes.append(150)
            elif health > 0.4:
                node_colors.append('#f59e0b')
                node_sizes.append(200)
            else:
                node_colors.append('#ef4444')
                node_sizes.append(250)
    
    # Dibujar red
    for edge in G.edges(data=True):
        weight = edge[2].get('weight', 0.5)
        edge_widths.append(weight * 2)
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=edge_widths, edge_color='#64748b')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.85, edgecolors='#475569', linewidths=1.5)
    
    ax.axis('off')
    ax.set_title(f'Agent Network Topology (t={step_counter.value})\nGreen: Healthy | Orange: Stressed | Red: Critical | Black: Liquidated', 
                fontsize=10, pad=10)
    
    fig.tight_layout()
    
    with solara.Card("Network Structure & Contagion Pathways", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


@solara.component
def PlotPhaseSpace():
    """Espacio de fases: Volatilidad vs Precio"""
    history = agent_states_history.value
    
    if len(history) < 2:
        return
    
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    prices = [h['price'] for h in history]
    volatilities = [h['volatility'] * 100 for h in history]
    steps = [h['step'] for h in history]
    
    # Gradient de color por tiempo
    scatter = ax.scatter(prices, volatilities, c=steps, cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Trayectoria
    ax.plot(prices, volatilities, 'gray', alpha=0.3, linewidth=1, linestyle='--')
    
    # Marcar shock
    if shock_step.value < len(history):
        shock_idx = shock_step.value
        ax.scatter(prices[shock_idx], volatilities[shock_idx], 
                  color='red', s=300, marker='*', edgecolors='darkred', 
                  linewidth=2, zorder=5, label='Shock Event')
    
    # Marcar inicio y fin
    ax.scatter(prices[0], volatilities[0], color='green', s=150, 
              marker='o', edgecolors='darkgreen', linewidth=2, 
              zorder=5, label='Start')
    ax.scatter(prices[-1], volatilities[-1], color='blue', s=150, 
              marker='s', edgecolors='darkblue', linewidth=2, 
              zorder=5, label='Current')
    
    ax.set_xlabel('Underlying Price ($)', fontsize=10)
    ax.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax.set_title('Phase Space Trajectory', fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=8)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time Step', fontsize=9)
    
    fig.tight_layout()
    
    with solara.Card("Phase Space Analysis", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


@solara.component
def PlotLiquidationCascades():
    """Cascadas de liquidación"""
    history = agent_states_history.value
    
    if len(history) < 2:
        return
    
    fig = Figure(figsize=(7, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Calcular métricas de cascada
    liquidated_count = []
    stress_by_type = {'Market Maker': [], 'Speculator': [], 'Hedger': [], 'Leveraged': []}
    
    for record in history:
        liquidated = sum(1 for s in record['states'] if s['liquidated'])
        liquidated_count.append(liquidated)
        
        # Estrés promedio por tipo
        for agent_type in stress_by_type.keys():
            agents_of_type = [s for s in record['states'] if s['type'] == agent_type and not s['liquidated']]
            if agents_of_type:
                avg_stress = np.mean([s['stress'] for s in agents_of_type])
                stress_by_type[agent_type].append(avg_stress)
            else:
                stress_by_type[agent_type].append(np.nan)
    
    steps = range(len(history))
    
    # Gráfico 1: Liquidaciones acumuladas
    ax1.fill_between(steps, 0, liquidated_count, color='#ef4444', alpha=0.6)
    ax1.plot(steps, liquidated_count, color='#991b1b', linewidth=2)
    
    if shock_step.value < len(history):
        ax1.axvline(x=shock_step.value, color='black', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Shock')
    
    ax1.set_ylabel('Cumulative Liquidations', fontsize=9)
    ax1.set_title('Liquidation Cascade Dynamics', fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Gráfico 2: Estrés por tipo de agente
    colors = {'Market Maker': '#3b82f6', 'Speculator': '#10b981', 
             'Hedger': '#f59e0b', 'Leveraged': '#ef4444'}
    
    for agent_type, stress_values in stress_by_type.items():
        ax2.plot(steps, stress_values, label=agent_type, 
                color=colors[agent_type], linewidth=2, alpha=0.8)
    
    if shock_step.value < len(history):
        ax2.axvline(x=shock_step.value, color='black', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Average Stress Level', fontsize=9)
    ax2.set_title('Stress Evolution by Agent Type', fontsize=10, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=7, ncol=2)
    ax2.set_ylim([0, 1])
    
    fig.tight_layout()
    
    with solara.Card("Liquidation Cascades & Agent Stress", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


@solara.component
def PlotAgentDistributions():
    """Distribuciones de salud y capital de agentes"""
    history = agent_states_history.value
    
    if len(history) == 0:
        return
    
    current_states = history[-1]['states']
    
    fig = Figure(figsize=(7, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Distribución de salud
    health_values = [s['health'] for s in current_states if not s['liquidated']]
    
    if len(health_values) > 0:
        ax1.hist(health_values, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax1.axvline(x=np.mean(health_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(health_values):.2f}')
        ax1.axvline(x=np.median(health_values), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(health_values):.2f}')
    
    ax1.set_xlabel('Health Ratio (Capital/Initial)', fontsize=9)
    ax1.set_ylabel('Frequency', fontsize=9)
    ax1.set_title('Agent Health Distribution', fontsize=10, weight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Distribución de capital por tipo
    type_colors = {'Market Maker': '#3b82f6', 'Speculator': '#10b981', 
                  'Hedger': '#f59e0b', 'Leveraged': '#ef4444'}
    
    types = list(set(s['type'] for s in current_states))
    positions = np.arange(len(types))
    
    capitals = []
    for agent_type in types:
        agents_of_type = [s for s in current_states if s['type'] == agent_type and not s['liquidated']]
        if agents_of_type:
            avg_capital = np.mean([s['capital'] for s in agents_of_type])
            capitals.append(avg_capital)
        else:
            capitals.append(0)
    
    bars = ax2.bar(positions, capitals, color=[type_colors.get(t, '#64748b') for t in types],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(types, rotation=15, ha='right', fontsize=8)
    ax2.set_ylabel('Average Capital ($)', fontsize=9)
    ax2.set_title('Capital Distribution by Agent Type', fontsize=10, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, capitals)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=7)
    
    fig.tight_layout()
    
    with solara.Card("Statistical Distributions", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


@solara.component
def PlotSpectralAnalysis():
    """Análisis espectral de volatilidad"""
    data = model_data.value
    
    if len(data) < 20:
        return
    
    fig = Figure(figsize=(7, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Serie temporal de volatilidad
    volatility = data['Implied_Volatility'].values * 100
    
    ax1.plot(data.index, volatility, color='#f59e0b', linewidth=2)
    ax1.fill_between(data.index, volatility, alpha=0.3, color='#f59e0b')
    
    if shock_step.value in data.index:
        ax1.axvline(x=shock_step.value, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Shock')
    
    ax1.set_ylabel('Implied Volatility (%)', fontsize=9)
    ax1.set_title('Volatility Time Series', fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Análisis de frecuencias (FFT)
    if len(volatility) > 10:
        # Detrend
        detrended = volatility - np.mean(volatility)
        
        # FFT
        fft_vals = np.fft.fft(detrended)
        fft_freq = np.fft.fftfreq(len(detrended))
        
        # Solo frecuencias positivas
        positive_freq_idx = fft_freq > 0
        frequencies = fft_freq[positive_freq_idx]
        power = np.abs(fft_vals[positive_freq_idx])**2
        
        ax2.plot(frequencies, power, color='#8b5cf6', linewidth=2)
        ax2.fill_between(frequencies, power, alpha=0.3, color='#8b5cf6')
        
        ax2.set_xlabel('Frequency', fontsize=9)
        ax2.set_ylabel('Power Spectrum', fontsize=9)
        ax2.set_title('Frequency Domain Analysis (FFT)', fontsize=10, weight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 0.5])
    
    fig.tight_layout()
    
    with solara.Card("Spectral Analysis of Volatility", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


@solara.component
def PlotContagionMetrics():
    """Métricas de contagio"""
    history = agent_states_history.value
    
    if len(history) < 2:
        return
    
    fig = Figure(figsize=(7, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Calcular métricas
    contagion_index = []
    network_fragility = []
    
    for record in history:
        states = record['states']
        
        # Índice de contagio: proporción de agentes en riesgo o críticos
        at_risk = sum(1 for s in states if s['health'] < 0.7 and not s['liquidated'])
        contagion_index.append(at_risk / len(states))
        
        # Fragilidad de red: varianza del estrés
        stress_values = [s['stress'] for s in states if not s['liquidated']]
        if len(stress_values) > 1:
            network_fragility.append(np.std(stress_values))
        else:
            network_fragility.append(0)
    
    steps = range(len(history))
    
    # Gráfico 1: Índice de contagio
    ax1.plot(steps, contagion_index, color='#ef4444', linewidth=2.5, label='Contagion Index')
    ax1.fill_between(steps, 0, contagion_index, color='#ef4444', alpha=0.3)
    
    if shock_step.value < len(history):
        ax1.axvline(x=shock_step.value, color='black', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax1.set_ylabel('Proportion of Stressed Agents', fontsize=9)
    ax1.set_title('Systemic Contagion Index', fontsize=10, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Gráfico 2: Fragilidad de red
    ax2.plot(steps, network_fragility, color='#8b5cf6', linewidth=2.5, label='Network Fragility')
    ax2.fill_between(steps, 0, network_fragility, color='#8b5cf6', alpha=0.3)
    
    if shock_step.value < len(history):
        ax2.axvline(x=shock_step.value, color='black', linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Stress Variance (σ)', fontsize=9)
    ax2.set_title('Network Fragility Measure', fontsize=10, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    with solara.Card("Contagion & Systemic Risk Metrics", style={"margin": "10px"}):
        solara.FigureMatplotlib(fig)
    
    plt.close(fig)


# Punto de entrada
Page = DiffusionVisualizationApp