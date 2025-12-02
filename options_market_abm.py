"""
Modelo ABM: Mercado de Opciones en Crisis de Liquidez
Implementación en Mesa 3.x con proceso de difusión (GBM) para el activo subyacente
"""


import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CLASES DE AGENTES
# ==============================================================================

class MarketMaker(Agent):
    """Market Makers: Proveedores de liquidez"""
    
    def __init__(self, model, capital, spread_base, risk_aversion):
        super().__init__(model)
        self.capital = capital
        self.initial_capital = capital
        self.spread_base = spread_base
        self.risk_aversion = risk_aversion
        self.inventory = 0  # Inventario de opciones (+ long, - short)
        self.max_inventory = 50
        
    def step(self):
        """Ajustar spreads según inventario y volatilidad"""
        # Ajuste de spread por inventario
        inventory_factor = 1 + abs(self.inventory) / self.max_inventory
        
        # Ajuste de spread por volatilidad
        volatility_factor = 1 + (self.model.current_volatility / 0.20 - 1) * self.risk_aversion
        
        # Spread ajustado
        self.current_spread = self.spread_base * inventory_factor * volatility_factor
        
        # En crisis, ampliar spreads dramáticamente
        if self.model.in_crisis:
            self.current_spread *= 2.0
    
    def provide_liquidity(self):
        """Proporcionar bid/ask al mercado"""
        mid_price = self.model.option_price
        half_spread = mid_price * self.current_spread / 2
        
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        
        return {'bid': max(0.01, bid), 'ask': ask, 'size': 10}


class Speculator(Agent):
    """Especuladores con diferentes estrategias"""
    
    def __init__(self, model, capital, strategy, signal_threshold):
        super().__init__(model)
        self.capital = capital
        self.initial_capital = capital
        self.strategy = strategy  # 'momentum' o 'mean_reversion'
        self.signal_threshold = signal_threshold
        self.position = 0
        self.max_position = int(capital / 1000)
        
    def step(self):
        """Operar según señales técnicas"""
        price_return = (self.model.underlying_price / self.model.prev_underlying_price - 1)
        
        if self.strategy == 'momentum':
            # Comprar en tendencia alcista
            if price_return > self.signal_threshold and self.position < self.max_position:
                self.buy_signal()
            # Vender en tendencia bajista
            elif price_return < -self.signal_threshold and self.position > -self.max_position:
                self.sell_signal()
                
        elif self.strategy == 'mean_reversion':
            # Vender cuando precio sube mucho (espera reversión)
            if price_return > self.signal_threshold and self.position > -self.max_position:
                self.sell_signal()
            # Comprar cuando precio baja mucho (espera reversión)
            elif price_return < -self.signal_threshold and self.position < self.max_position:
                self.buy_signal()
    
    def buy_signal(self):
        """Generar señal de compra"""
        self.position += 1
        cost = self.model.option_price
        self.capital -= cost
        
    def sell_signal(self):
        """Generar señal de venta"""
        self.position -= 1
        proceeds = self.model.option_price
        self.capital += proceeds


class Hedger(Agent):
    """Hedgers institucionales cubriendo portafolios"""
    
    def __init__(self, model, stock_portfolio_value, hedge_ratio, rebalance_freq):
        super().__init__(model)
        self.stock_portfolio_value = stock_portfolio_value
        self.hedge_ratio = hedge_ratio  # % del portafolio a cubrir
        self.rebalance_freq = rebalance_freq
        self.options_position = 0
        self.steps_since_rebalance = 0
        
    def step(self):
        """Rebalancear cobertura delta periódicamente"""
        self.steps_since_rebalance += 1
        
        # Aumentar frecuencia de rebalanceo en crisis
        freq = self.rebalance_freq // 2 if self.model.in_crisis else self.rebalance_freq
        
        if self.steps_since_rebalance >= freq:
            self.rebalance_hedge()
            self.steps_since_rebalance = 0
            
        # Aumentar cobertura en crisis
        if self.model.in_crisis and self.hedge_ratio < 0.9:
            self.hedge_ratio = min(0.9, self.hedge_ratio * 1.2)
    
    def rebalance_hedge(self):
        """Ajustar posición de opciones para mantener delta neutral"""
        # Evitar división por cero o precios muy pequeños
        if self.model.option_price < 0.01:
            return
        
        target_options = int(self.stock_portfolio_value * self.hedge_ratio / self.model.option_price)
        # Limitar tamaño de la posición
        target_options = min(target_options, 100000)
        self.options_position = target_options


class Leveraged(Agent):
    """Traders apalancados susceptibles a margin calls"""
    
    def __init__(self, model, own_capital, leverage, margin_requirement):
        super().__init__(model)
        self.own_capital = own_capital
        self.initial_capital = own_capital
        self.leverage = leverage
        self.margin_requirement = margin_requirement  # % del valor de la posición
        self.position = 0
        self.borrowed = 0
        self.liquidated = False
        
    def step(self):
        """Monitorizar margen y operar con apalancamiento"""
        if self.liquidated:
            return
        
        # Calcular valor actual de la posición
        position_value = self.position * self.model.option_price
        
        # Calcular capital disponible
        total_assets = self.own_capital + position_value - self.borrowed
        
        # Verificar margin call
        if self.position != 0:
            required_margin = abs(position_value) * self.margin_requirement
            
            if total_assets < required_margin:
                self.trigger_margin_call()
                return
        
        # Si no está liquidado, puede operar
        if not self.liquidated and self.model.step_num % 5 == 0:
            self.speculative_trade()
    
    def speculative_trade(self):
        """Realizar trade especulativo con apalancamiento"""
        available_to_borrow = self.own_capital * (self.leverage - 1)
        
        # Tomar posición si tiene capital
        if self.own_capital > 100 and available_to_borrow > 0:
            trade_size = int(available_to_borrow / self.model.option_price / 10)
            if trade_size > 0:
                # 50% probabilidad de long o short
                direction = 1 if self.random.random() > 0.5 else -1
                self.position += direction * trade_size
                self.borrowed += trade_size * self.model.option_price
    
    def trigger_margin_call(self):
        """Ejecutar liquidación forzada"""
        self.liquidated = True
        self.model.margin_calls += 1
        
        # Liquidar posición (venta forzada impacta el mercado)
        liquidation_loss = abs(self.position) * self.model.option_price * 0.1  # 10% de pérdida por liquidación
        self.own_capital = max(0, self.own_capital - liquidation_loss)
        
        self.position = 0
        self.borrowed = 0


# ==============================================================================
# MODELO PRINCIPAL
# ==============================================================================

class OptionsMarketModel(Model):
    """Modelo ABM de Mercado de Opciones con Crisis de Liquidez"""
    
    def __init__(self, 
                 n_market_makers=7,
                 n_speculators=40,
                 n_hedgers=15,
                 n_leveraged=25,
                 initial_underlying_price=100.0,
                 initial_volatility=0.20,
                 shock_step=50,
                 shock_magnitude=-0.10,
                 seed=None):
        
        super().__init__(seed=seed)
        
        # Parámetros del modelo
        self.n_market_makers = n_market_makers
        self.n_speculators = n_speculators
        self.n_hedgers = n_hedgers
        self.n_leveraged = n_leveraged
        
        # Precio del activo subyacente (proceso de difusión GBM)
        self.underlying_price = initial_underlying_price
        self.prev_underlying_price = initial_underlying_price
        self.initial_underlying_price = initial_underlying_price
        self.volatility = initial_volatility
        self.current_volatility = initial_volatility
        self.mu = 0.0  # Drift esperado (neutral)
        
        # Shock
        self.shock_step = shock_step
        self.shock_magnitude = shock_magnitude
        self.shock_applied = False
        
        # Estado del mercado
        self.in_crisis = False
        self.margin_calls = 0
        self.step_num = 0
        
        # Precio de la opción (call ATM, simplificado)
        self.strike_price = initial_underlying_price
        self.time_to_maturity = 30 / 365  # 30 días
        self.option_price = self.black_scholes_call(
            self.underlying_price, 
            self.strike_price, 
            self.time_to_maturity, 
            0.02,  # risk-free rate
            self.volatility
        )
        
        # Order book simplificado
        self.order_book = {'bids': [], 'asks': []}
        
        # Crear agentes
        self.create_agents()
        
        # DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Underlying_Price": "underlying_price",
                "Option_Price": "option_price",
                "Implied_Volatility": "current_volatility",
                "Margin_Calls": "margin_calls",
                "Bid_Ask_Spread": self.compute_spread,
                "Liquidity_Index": self.compute_liquidity,
                "In_Crisis": "in_crisis",
                "MM_Avg_Capital": lambda m: self.avg_capital_by_type(MarketMaker),
                "Speculator_Avg_Capital": lambda m: self.avg_capital_by_type(Speculator),
                "Leveraged_Active": lambda m: sum(1 for a in m.agents if isinstance(a, Leveraged) and not a.liquidated)
            }
        )
    
    def create_agents(self):
        """Crear todos los agentes del mercado"""
        
        # Market Makers
        for _ in range(self.n_market_makers):
            capital = self.random.uniform(100_000, 500_000)
            spread = self.random.uniform(0.005, 0.020)
            risk_aversion = self.random.uniform(0.5, 2.0)
            agent = MarketMaker(self, capital, spread, risk_aversion)
            self.agents.add(agent)
        
        # Especuladores
        for _ in range(self.n_speculators):
            capital = self.random.uniform(10_000, 100_000)
            strategy = self.random.choice(['momentum', 'mean_reversion'])
            signal_threshold = self.random.uniform(0.01, 0.03)
            agent = Speculator(self, capital, strategy, signal_threshold)
            self.agents.add(agent)
        
        # Hedgers
        for _ in range(self.n_hedgers):
            portfolio_value = self.random.uniform(500_000, 2_000_000)
            hedge_ratio = self.random.uniform(0.3, 0.7)
            rebalance_freq = int(self.random.uniform(5, 15))
            agent = Hedger(self, portfolio_value, hedge_ratio, rebalance_freq)
            self.agents.add(agent)
        
        # Apalancados
        for _ in range(self.n_leveraged):
            capital = self.random.uniform(5_000, 50_000)
            leverage = self.random.uniform(2.0, 5.0)
            margin_req = self.random.uniform(0.25, 0.40)
            agent = Leveraged(self, capital, leverage, margin_req)
            self.agents.add(agent)
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Fórmula de Black-Scholes para precio de call"""
        if T <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def update_underlying_price_gbm(self):
        """Actualizar precio del activo mediante Geometric Brownian Motion (proceso de difusión)"""
        dt = 1/252  # 1 día de trading
        
        # Aplicar shock si corresponde
        if self.step_num == self.shock_step and not self.shock_applied:
            shock_return = self.shock_magnitude
            self.shock_applied = True
            self.in_crisis = True
        else:
            # GBM: dS = μS dt + σS dW
            drift = self.mu * dt
            # Usar gauss() en lugar de normal() para compatibilidad con Mesa 3.x
            diffusion = self.volatility * np.sqrt(dt) * self.random.gauss(0, 1)
            shock_return = drift + diffusion
        
        # Actualizar precio
        self.prev_underlying_price = self.underlying_price
        self.underlying_price = self.underlying_price * (1 + shock_return)
        
        # Volatilidad implícita aumenta en crisis
        if self.in_crisis:
            self.current_volatility = min(0.80, self.volatility * (1 + abs(shock_return) * 10))
        else:
            self.current_volatility = self.volatility
        
        # Actualizar precio de la opción
        self.option_price = self.black_scholes_call(
            self.underlying_price,
            self.strike_price,
            self.time_to_maturity,
            0.02,
            self.current_volatility
        )
    
    def update_order_book(self):
        """Actualizar order book con quotes de market makers"""
        self.order_book = {'bids': [], 'asks': []}
        
        for agent in self.agents:
            if isinstance(agent, MarketMaker):
                quote = agent.provide_liquidity()
                self.order_book['bids'].append(quote['bid'])
                self.order_book['asks'].append(quote['ask'])
    
    def compute_spread(self):
        """Calcular spread promedio del mercado"""
        if self.order_book['bids'] and self.order_book['asks']:
            best_bid = max(self.order_book['bids'])
            best_ask = min(self.order_book['asks'])
            return (best_ask - best_bid) / self.option_price
        return 0.0
    
    def compute_liquidity(self):
        """Índice de liquidez (inverso del spread normalizado)"""
        spread = self.compute_spread()
        if spread > 0:
            return 1 / (1 + spread * 100)  # Normalizado
        return 1.0
    
    def avg_capital_by_type(self, agent_type):
        """Capital promedio por tipo de agente"""
        agents = [a for a in self.agents if isinstance(a, agent_type)]
        if not agents:
            return 0
        if agent_type == MarketMaker or agent_type == Speculator or agent_type == Leveraged:
            return np.mean([a.capital for a in agents])
        return 0
    
    def step(self):
        """Avanzar un paso en la simulación"""
        self.step_num += 1
        
        # 1. Actualizar precio del subyacente (proceso de difusión GBM)
        self.update_underlying_price_gbm()
        
        # 2. Activar agentes en orden aleatorio
        self.agents.shuffle_do("step")
        
        # 3. Actualizar order book
        self.update_order_book()
        
        # 4. Detectar crisis (múltiples margin calls)
        if self.margin_calls > 5:
            self.in_crisis = True
        
        # 5. Recolectar datos
        self.datacollector.collect(self)
        
        # Reducir tiempo hasta vencimiento
        self.time_to_maturity = max(0.001, self.time_to_maturity - 1/365)


# ==============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR SIMULACIÓN
# ==============================================================================

def run_simulation(steps=200, seed=42):
    """Ejecutar simulación del mercado de opciones"""
    
    print("=" * 70)
    print("SIMULACIÓN: Mercado de Opciones en Crisis de Liquidez")
    print("Modelo ABM con Proceso de Difusión (GBM)")
    print("=" * 70)
    
    # Crear modelo
    model = OptionsMarketModel(
        n_market_makers=7,
        n_speculators=40,
        n_hedgers=15,
        n_leveraged=25,
        initial_underlying_price=100.0,
        initial_volatility=0.20,
        shock_step=50,
        shock_magnitude=-0.10,
        seed=seed
    )
    
    # Ejecutar simulación
    print(f"\nEjecutando {steps} pasos de simulación...")
    print(f"Shock de precio (-10%) aplicado en el paso {model.shock_step}\n")
    
    for i in range(steps):
        model.step()
        
        if i % 50 == 0:
            print(f"Paso {i:3d} | Precio Subyacente: ${model.underlying_price:7.2f} | "
                  f"Precio Opción: ${model.option_price:6.2f} | "
                  f"Volatilidad: {model.current_volatility*100:5.1f}% | "
                  f"Margin Calls: {model.margin_calls:3d}")
    
    # Obtener resultados
    results = model.datacollector.get_model_vars_dataframe()
    
    print("\n" + "=" * 70)
    print("RESULTADOS DE LA SIMULACIÓN")
    print("=" * 70)
    print(f"\nEstadísticas Finales:")
    print(f"  Precio Final Subyacente: ${model.underlying_price:.2f} "
          f"(Cambio: {(model.underlying_price/model.initial_underlying_price - 1)*100:+.2f}%)")
    print(f"  Precio Final Opción: ${model.option_price:.2f}")
    print(f"  Volatilidad Implícita Final: {model.current_volatility*100:.1f}%")
    print(f"  Total Margin Calls: {model.margin_calls}")
    print(f"  Spread Promedio Final: {results['Bid_Ask_Spread'].iloc[-1]*100:.2f}%")
    print(f"  Índice de Liquidez Final: {results['Liquidity_Index'].iloc[-1]:.3f}")
    
    print(f"\n  Capital Promedio Market Makers: ${results['MM_Avg_Capital'].iloc[-1]:,.0f}")
    print(f"  Capital Promedio Especuladores: ${results['Speculator_Avg_Capital'].iloc[-1]:,.0f}")
    print(f"  Traders Apalancados Activos: {int(results['Leveraged_Active'].iloc[-1])}/{model.n_leveraged}")
    
    return model, results


# ==============================================================================
# VISUALIZACIÓN Y ANÁLISIS
# ==============================================================================

def plot_results(results, output_file='market_simulation_results.png'):
    """Generar gráficos de los resultados"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Mercado de Opciones - Crisis de Liquidez (Modelo ABM con Difusión GBM)', 
                 fontsize=14, fontweight='bold')
    
    # Gráfico 1: Precio del Subyacente
    axes[0, 0].plot(results.index, results['Underlying_Price'], 'b-', linewidth=2)
    axes[0, 0].axvline(x=50, color='r', linestyle='--', label='Shock Aplicado')
    axes[0, 0].set_title('Precio del Activo Subyacente (GBM)')
    axes[0, 0].set_xlabel('Paso')
    axes[0, 0].set_ylabel('Precio ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Precio de la Opción
    axes[0, 1].plot(results.index, results['Option_Price'], 'g-', linewidth=2)
    axes[0, 1].axvline(x=50, color='r', linestyle='--', label='Shock Aplicado')
    axes[0, 1].set_title('Precio de la Opción (Black-Scholes)')
    axes[0, 1].set_xlabel('Paso')
    axes[0, 1].set_ylabel('Precio ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Volatilidad Implícita
    axes[1, 0].plot(results.index, results['Implied_Volatility']*100, 'orange', linewidth=2)
    axes[1, 0].axvline(x=50, color='r', linestyle='--', label='Shock Aplicado')
    axes[1, 0].set_title('Volatilidad Implícita')
    axes[1, 0].set_xlabel('Paso')
    axes[1, 0].set_ylabel('Volatilidad (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Margin Calls Acumulados
    axes[1, 1].plot(results.index, results['Margin_Calls'], 'r-', linewidth=2)
    axes[1, 1].axvline(x=50, color='r', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Margin Calls Acumulados')
    axes[1, 1].set_xlabel('Paso')
    axes[1, 1].set_ylabel('Cantidad')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Gráfico 5: Bid-Ask Spread
    axes[2, 0].plot(results.index, results['Bid_Ask_Spread']*100, 'purple', linewidth=2)
    axes[2, 0].axvline(x=50, color='r', linestyle='--', label='Shock Aplicado')
    axes[2, 0].set_title('Bid-Ask Spread')
    axes[2, 0].set_xlabel('Paso')
    axes[2, 0].set_ylabel('Spread (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Gráfico 6: Índice de Liquidez
    axes[2, 1].plot(results.index, results['Liquidity_Index'], 'teal', linewidth=2)
    axes[2, 1].axvline(x=50, color='r', linestyle='--', label='Shock Aplicado')
    axes[2, 1].set_title('Índice de Liquidez')
    axes[2, 1].set_xlabel('Paso')
    axes[2, 1].set_ylabel('Índice (0-1)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráficos guardados en: {output_file}")
    plt.close()


def export_results(results, output_file='simulation_data.csv'):
    """Exportar resultados a CSV"""
    results.to_csv(output_file, index=True)
    print(f"✓ Datos exportados a: {output_file}")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # Ejecutar simulación
    model, results = run_simulation(steps=200, seed=42)
    
    # Generar visualizaciones
    plot_results(results, output_file=r'C:\Users\japal\Downloads\Juegos\outputs\market_simulation_results.png')
    
    # Exportar datos
    export_results(results, output_file=r'C:\Users\japal\Downloads\Juegos\outputs\simulation_data.csv')
    
    print("\n" + "=" * 70)
    print("✓ SIMULACIÓN COMPLETADA")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  1. market_simulation_results.png - Gráficos de análisis")
    print("  2. simulation_data.csv - Datos completos de la simulación")
    print("\nEl modelo implementa:")
    print("  • Proceso de difusión GBM para el precio del activo subyacente")
    print("  • 4 tipos de agentes con comportamientos heterogéneos")
    print("  • Sistema de márgenes y liquidaciones forzadas")
    print("  • Cascadas de margin calls amplificando volatilidad")
    print("  • Métricas de liquidez y spreads dinámicos")