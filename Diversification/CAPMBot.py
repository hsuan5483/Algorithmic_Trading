"""
This is a template bot for  the CAPM Task.
"""
from enum import Enum
from typing import List, Dict
from fmclient import Agent, Session
from fmclient import Order, OrderSide, OrderType
import copy
import pandas as pd
import numpy as np
from scipy import interpolate
from itertools import permutations

# Submission details
SUBMISSION = {"student_number": "1424430", "name": "Pei-Hsuan (Amelia) Hsu"}

FM_ACCOUNT = "mickle-coequal"
FM_EMAIL = "YOUR EMAIL"
FM_PASSWORD = "YOUR PASSWORD"
MARKETPLACE_ID = 1444 # replace this with the marketplace id


class CAPMBot(Agent):

    def __init__(self, account, email, password, marketplace_id, risk_penalty=0.001, session_time=20):
        """
        Constructor for the Bot
        :param account: Account name
        :param email: Email id
        :param password: password
        :param marketplace_id: id of the marketplace
        :param risk_penalty: Penalty for risk
        :param session_time: Total trading time for one session
        """
        super().__init__(account, email, password, marketplace_id, name="CAPM Bot")
        self._payoffs = {}
        self._risk_penalty = risk_penalty
        self._session_time = session_time
        self._market_ids = {}
                
        # store current sent order in each mode
        self._sent_reactive_order = None
        self._sent_proactive_order = None

        # scale units
        self._unit_convertor = 1/100 # to dollar

        # mean for each security
        self._mean = {}

        # variance for each security
        self._variance = {}

        # covariance for each pair of securities
        self._covariance = {}

        # holdings after trads
        self._current_holdings_units = {}

        # get best bid/ask price: Dict[str:Order]
        self._current_best_market_orders = {}

        # list of trial orders
        self._trial_orders = None

        # count refresh time
        self._refresh_count = 0
        

    def initialised(self):
        # Extract payoff distribution for each security
        for market_id, market_info in self.markets.items():
            security = market_info.item
            description = market_info.description
            self._payoffs[security] = [int(a)*self._unit_convertor for a in description.split(",")] # convert unit into dollars
            self._market_ids[security] = market_info
            self.inform(f"market: {market_id} / {market_info.name} / {market_info.unit_tick}")
            self.inform(f"security: {security} / description: {description}")
            
            n = len(self._payoffs[security])

            # calculate mean for each security
            self._mean[security] = sum(self._payoffs[security]) * (1/n)

            # calculate variance for each security
            self._variance[security] = sum(np.power(self._payoffs[security],2)) * (1/n) - pow(sum(self._payoffs[security]), 2) * (1/pow(n,2))

        # calculate covariance between different securities
        perms = permutations(list(self._payoffs.keys()), 2)
        for perm in perms:
            self._covariance[perm] = np.dot(self._payoffs[perm[0]], self._payoffs[perm[1]]) * (1/n) - np.mean(self._payoffs[perm[0]]) * np.mean(self._payoffs[perm[1]])

        self.inform("Bot initialised, I have the payoffs for the states.")

    def get_potential_performance(self, orders: List[Order]=[]):
        """
        Returns the portfolio performance if the given list of orders is executed.
        The performance as per the following formula:
        Performance = ExpectedPayoff - b * PayoffVariance, where b is the penalty for risk.
        :param orders: list of orders
        :return:
        """
                
        try:

            # get holdings after trades
            new_cash_available, new_holdings_units = self._get_holdings_after_trades(orders)
            
            # calculate expected payoff
            exp_payoff = self._get_expected_payoff(new_cash_available, new_holdings_units)
            
            # calculate payoff variance
            payoff_variance = self._get_payoff_variance(new_holdings_units)
        
        except Exception as e:
            self.error(e)
        
        return exp_payoff - self._risk_penalty * payoff_variance
    
    def _get_holdings_after_trades(self, orders:List[Order]=[]):
        # function for calculating holdings after trades
        new_holdings_units = self._current_holdings_units.copy()
        new_cash_available = self._holdings.cash_available
        for order in orders:
            if order.order_side == OrderSide.BUY:
                new_holdings_units[order.market.item] += order.units
                new_cash_available -= order.price
            else:
                new_holdings_units[order.market.item] -= order.units
                new_cash_available += order.price
        return new_cash_available, new_holdings_units

    def _get_expected_payoff(self, cash=None, holdings_units=None):
        if cash is None:
            cash = self.holdings.cash_available

        if holdings_units is None:
            holdings_units = self._current_holdings_units.copy()

        return cash * self._unit_convertor + np.dot(np.array(list(holdings_units.values())), np.array(list(self._mean.values())))
    
    def _get_payoff_variance(self, holdings_units=None):
        if holdings_units is None:
            holdings_units = self._current_holdings_units.copy()

        payoff_variance = 0
        for i in list(self._payoffs.keys()):
            for j in list(self._payoffs.keys()):
                if i == j:
                    payoff_variance += pow(holdings_units[i],2) * self._variance[i]
                else:
                    payoff_variance += holdings_units[i] * holdings_units[j] * self._covariance[(i, j)]
        
        return payoff_variance

    def _get_sharpe_ratio(self, orders:List[Order]=[]):
        cash, holdings_units = self._get_holdings_after_trades(orders)
        exp_payoff = self._get_expected_payoff(cash, holdings_units)
        payoff_variance = self._get_payoff_variance(holdings_units)
        return exp_payoff / np.sqrt(payoff_variance)

    def is_portfolio_optimal(self):
        """
        Returns true if the current holdings are optimal (as per the performance formula), false otherwise.
        :return:
        """

        # current market orders by security
        current_orders = self._get_market_orders()
        
        
        # get bid/ask price for each market
        for security in current_orders.keys():
            self._current_best_market_orders[security] = self._get_best_market_orders(current_orders[security])

        # create orders for placing opposite position orders to the current market orders
        opposite_orders = []
        for security in current_orders.keys():
            for order in self._current_best_market_orders[security]:
                if order is not None:
                    if order.order_side == OrderSide.BUY:
                        if self._current_holdings_units[security] > 1:
                            new_order = self._set_order(self._market_ids[security], order.order_type, OrderSide.SELL, order.price, 1, 'reactive')
                            opposite_orders.append(new_order)
                    else:
                        if self._holdings.cash_available > order.price:
                            new_order = self._set_order(self._market_ids[security], order.order_type, OrderSide.BUY, order.price, 1, 'reactive')
                            opposite_orders.append(new_order)

        # get the dict of orders with improvements on performance that could improve the portfolio
        order_improvements = self._improvements_per_order(opposite_orders)

        # if there is any order in the market can improve the portfolio, store the order that can improve the portfolio the most
        if bool(order_improvements):
            max_improvement = max(order_improvements.keys())
            self._sent_reactive_order = order_improvements[max_improvement]

        # if the improvement dict is empty, return True; otherwise, return False
        return not order_improvements
    

    def _get_best_market_orders(self, orders: List[Order] = []):
                
        if not orders:
            return []
        
        else:
            bid = None
            ask = None
            best_bid_order = None
            best_ask_order = None

            for order in orders:

                if (bid is None) & (order.order_side == OrderSide.BUY):
                    bid = order.price
                    best_bid_order = order
                
                if (ask is None) & (order.order_side == OrderSide.SELL):
                    ask = order.price
                    best_ask_order = order

                if order.order_side == OrderSide.BUY:
                    if order.price > bid:
                        bid = order.price
                        best_bid_order = order
                else:
                    if order.price < ask:
                        ask = order.price
                        best_ask_order = order
            
            return [best_bid_order, best_ask_order]

    def _get_market_orders(self):
        
        # current placed orders
        current_orders = {x:[] for x in self._payoffs.keys()}
        
        try:
            # get current market orders
            for order_id, order in Order.current().items():
                if not order.mine:
                    current_orders[order.market.item].append(order)
        
        except Exception as e:
            self.error(e)

        return current_orders
    
    def _improvements_per_order(self, orders: List[Order], sharpe_base=False, filter=True):

        # current portfolio performance
        if sharpe_base:
            current_performance = self._get_sharpe_ratio()
        else:
            current_performance = self.get_potential_performance()

        # record orders improve the portfolio
        improvements = {}
        for order in orders:
            if sharpe_base:
                new_performance = self._get_sharpe_ratio([order])
            else:
                new_performance = self.get_potential_performance([order])
            
            
            score = new_performance - current_performance
            if filter:
                if score > 0:
                    improvements[score] = order
            else:
                improvements[score] = order

        return improvements

    
    def _find_best_order_to_place(self):

        if bool(self._trial_orders):
            if self._get_payoff_variance() != 0:
                # use Sharpe Ratio as target
                target = 0.005
                if self._refresh_count == 30:
                    target -= 0.001
                trial_orders_improvements = self._improvements_per_order(self._trial_orders, True, False)
            else:
                # use performance as target
                target = 3
                if self._refresh_count == 30:
                    target -= 0.5
                trial_orders_improvements = self._improvements_per_order(self._trial_orders, False, False)
            
        else:
            return None
        
        best_order = None
        highest_rate = -1
        securities = np.unique([x.market.item for x in trial_orders_improvements.values()])
        # security_improve_rates = {x:[] for x in securities}
        for security in securities:
            imp_rates = {OrderSide.BUY:{}, OrderSide.SELL:{}}
            for score, order in trial_orders_improvements.items():
                if order.market.item == security:
                    imp_rates[order.order_side][order.price] = score
            
            # current orders
            current_orders = self._current_best_market_orders[security]

            if None in current_orders:
                bid = None
                ask = None
            else:
                bid, ask = self._get_best_market_orders(current_orders)

            # calculate the rate of improvemet
            ## bid side
            x_temp = list(imp_rates[OrderSide.BUY].values()) # score (improvement)
            y_temp = list(imp_rates[OrderSide.BUY].keys()) # price
            
            if bool(x_temp) and bool(y_temp):
                f_buy = interpolate.interp1d(x_temp, y_temp)
                f_buy_inv = interpolate.interp1d(y_temp, x_temp)

                try:
                    if bid is not None and f_buy_inv(bid) > target:
                        p_buy = bid
                    else:
                        p_buy = int(f_buy(target))
                except:
                    p_buy = None

                # when using performance as target
                if self._get_payoff_variance() == 0:
                    while p_buy is None:
                        target -= 0.5
                        if target == 0:
                            break
                        try:
                            p_buy = int(f_buy(target))
                        except:
                            p_buy = None
                
                if p_buy is not None:
                    rate_temp = (y_temp[1] - y_temp[0]) / (x_temp[1] - x_temp[0])
                    if abs(rate_temp) > highest_rate:
                        highest_rate = rate_temp
                        best_order = self._set_order(self._market_ids[security], OrderType.LIMIT, OrderSide.BUY, p_buy, 1, 'proactive')

            ## ask side
            x_temp = list(imp_rates[OrderSide.SELL].values())
            y_temp = list(imp_rates[OrderSide.SELL].keys())

            if bool(x_temp) and bool(y_temp):
                f_sell = interpolate.interp1d(x_temp, y_temp)
                f_sell_inv = interpolate.interp1d(y_temp, x_temp)

                try:
                    if ask is not None and f_sell_inv(ask) > target:
                        p_sell = ask
                    else:
                        p_sell = int(f_sell(target))
                except:
                    p_sell = None
                
                if self._get_payoff_variance() == 0:
                    while p_sell is None:
                        target -= 0.1
                        if target == 0:
                            break
                        try:
                            p_sell = int(f_sell(target))
                        except:
                            p_sell = None
                
                if p_sell is not None:
                    rate_temp = (y_temp[1] - y_temp[0]) / (x_temp[1] - x_temp[0])
                    if abs(rate_temp) > highest_rate:
                        highest_rate = rate_temp
                        best_order = self._set_order(self._market_ids[security], OrderType.LIMIT, OrderSide.SELL, p_sell, 1, 'proactive')
        
        if self._sent_reactive_order is not None:
            if best_order is not None and self._sent_reactive_order.market != best_order.market:
                return best_order
            else:
                return None
        else:
            return best_order
            

    def _set_order(self, market, type, side, price, units, ref):
        
        new_order = Order.create_new(market)
        new_order.order_type = type
        new_order.order_side = side
        new_order.price = price
        new_order.units = units
        new_order.ref = ref
        
        return new_order

    def _check_holdings_status(self, order: Order):

        if order is None:
            return False
        
        else:
            if order.order_side == OrderSide.BUY:
                if self._holdings.cash_available > order.price:
                    return True
                else:
                    return False
            elif order.order_side == OrderSide.SELL:
                if self._current_holdings_units[order.market.item] > order.units:
                    return True
                else:
                    return False
    
    def _create_trial_orders(self):
        self._trial_orders = []
        for security, market in self._market_ids.items():
            if self.holdings.cash_available > 0:
                self._trial_orders.append(self._set_order(market, OrderType.LIMIT, OrderSide.BUY, 0, 1, 'proactive'))
                self._trial_orders.append(self._set_order(market, OrderType.LIMIT, OrderSide.BUY, 1000, 1, 'proactive'))
            if self._current_holdings_units[security] > 1:
                self._trial_orders.append(self._set_order(market, OrderType.LIMIT, OrderSide.SELL, 0, 1, 'proactive'))
                self._trial_orders.append(self._set_order(market, OrderType.LIMIT, OrderSide.SELL, 1000, 1, 'proactive'))
        
    def _cancel_order(self, order: Order):
        cancel_order = copy.copy(order)
        cancel_order.order_type = OrderType.CANCEL
        cancel_order.ref = "cancel-order"
        super().send_order(cancel_order)

    def _reactive_mode(self):
        self.inform("The reactive mode is activated...")
        
        if self._check_holdings_status(self._sent_reactive_order):
            super().send_order(self._sent_reactive_order)

    def _proactive_mode(self):
        if self._sent_proactive_order is None:
            self.inform("The proactive mode is activated...")
            self._sent_proactive_order = self._find_best_order_to_place()
            if self._check_holdings_status(self._sent_proactive_order):
                super().send_order(self._sent_proactive_order)
        
    def order_accepted(self, order):
        if self._get_payoff_variance() == 0:
            self.inform(f"The performance of the portfolio is improved. (Performance = {self.get_potential_performance():.2f} / Sharpe Ratio = NA)")
        else:
            self.inform(f"The performance of the portfolio is improved. (Performance = {self.get_potential_performance():.2f} / Sharpe Ratio = {self._get_sharpe_ratio():.2f})")
        
        if order.ref == 'reactive':
            if order.is_consumed:
                self._sent_reactive_order = None
        else:
            if order.is_consumed:
                self._sent_proactive_order = None

    def order_rejected(self, info, order):
        self.inform(f"Order was rejected: {order.ref}-mode order in {order.market.name} market / Error msg: {info['response']['error']}")

    def received_orders(self, orders: List[Order]):

        try:
            if self._sent_reactive_order is not None:
                if self._sent_reactive_order.is_consumed:
                    self._sent_reactive_order = None
            else:
                if not self.is_portfolio_optimal():
                    self._reactive_mode()
            
            if self._sent_proactive_order is not None:
                if self._sent_proactive_order.is_pending:
                    self._refresh_count += 1

                    if self._refresh_count > 30:
                        for order_id, order in Order.my_current().items():
                            if order.ref == 'proactive':
                                self._cancel_order(order)
                                self._sent_proactive_order = None
                                self._refresh_count = 0

            if self._sent_proactive_order is not None:
                if self._sent_proactive_order.is_consumed:
                    self._sent_proactive_order = None

            else:
                if self.is_portfolio_optimal():
                    self._create_trial_orders()
                    self._proactive_mode()
        
        except Exception as e:
            self.error(e)

    def received_session_info(self, session: Session):
        self.inform(f"session status: {session.is_open}")

    def pre_start_tasks(self):
        pass
        
    def received_holdings(self, holdings):
        self.inform(f"Cash: {holdings.cash}")
        self.inform(f"Cash available: {holdings.cash_available}")
        for market, asset in holdings.assets.items():
            self.inform(f"{market.name}: {asset.units_available}")
            self._current_holdings_units[market.item] = asset.units_available
            self.inform(f"{market.name}: {asset.units}")

if __name__ == "__main__":
    bot = CAPMBot(FM_ACCOUNT, FM_EMAIL, FM_PASSWORD, MARKETPLACE_ID)
    bot.run()
