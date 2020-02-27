''' gym environment for Rotman's RIT client '''
__author__ = 'Matthew Reiter'
__email__ = 'matthew.reiter@mail.utoronto.ca'
__version__ = '1.0.1'
__status__ = 'Production'
__copyright__   = 'Copyright 2020, Applications of Deep Reinforcement Learning, BASc Thesis, University of Toronto'

import gym
import time
import requests

import pandas as pd

s = requests.Session()

OH_COLS = ['order_id',          # unique identifier from the matching engine
           'i_placed',          # time index when order was placed on exchange
           'i_filled',          # time index when last filled, including partially so ... -1 if not filled
           'i_cancelled',       # time index when cancelled ... -1 if not cancelled
           'price',             # target price for limit order
           'quantity',          # target quantity for limit order
           'filled',            # quantity executed
           'vwap',              # running vwap for the order
           'active']            # False if not filled or cancelled

STATE_COLS = ['time',
              'start_time',
              'end_time',
              'inventory',
              'position',
              'pending',
              'cost',
              'vwap']


class RITEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    reward_signal = {'method': 'VWAP_TARGET',
                     'collection': 'harvested',
                     'target': 'beat'}

    actions = {'LMT': 0,
               'MKT': 1,
               'HLD': 2,
               'CNL': 3}
    params = {
        'ticker': 'MC',
        'api_key': 'SECRET_KEY',
        'port': '9999',
        'inventory': 100000,
        'direction': 1,
        'start_time': None,
        'end_time': None,
        'reward_signal': reward_signal,
    }

    def __init__(self, params):
        s.headers.update({'X-API-key': params['api_key']})
        self.api = 'http://localhost:' + params['port'] + '/v1/{}'

        self.ticker = params['ticker']
        self.trader_id = self._get_trader_id()

        # if we have a designated start time to begin trading
        self.status, self.time, max_time = self._get_tick_status()
        self.start_time = params['start_time'] if params['start_time'] is not None else 5
        self.end_time = params['end_time'] if params['end_time'] is not None else max_time - 10

        self.inventory, self.direction = params['inventory'], params['direction']

        self.position, self.cost = self._get_position()
        self.vwap_m = 0
        self.pending, self.pending_orders = self._get_pending()
        self.order_history = pd.DataFrame(columns=OH_COLS)

        self.delayed_reward = 0
        self.reward_signal = params['reward_signal']

    def reset(self):
        self.order_history = pd.DataFrame(columns=OH_COLS)
        self.delayed_reward = 0

        # wait until the simulator is live again
        self.status, self.time, _ = self._get_tick_status()
        while (self.status == 'ACTIVE' and abs(self.time - self.start_time) > 1) or (self.status == 'STOPPED'):
            self.status, self.time, _ = self._get_tick_status()

        # reset the position
        self.position, self.cost = self._get_position()
        self.vwap_m = self._calc_vwap()
        self.pending, self.pending_orders = self._get_pending()

        return self._get_state()

    def _get_state(self):
        # retrieve the order book
        order_book = self._get_lob()

        # construct the state information about the trader
        state_info = dict(zip(STATE_COLS, [self.time,
                                           self.start_time,
                                           self.end_time,
                                           self.inventory,
                                           self.position,
                                           self.pending,
                                           self.cost,
                                           self._calc_vwap()]))

        return order_book, state_info

    def step(self, action):
        # update the time and status
        self.status, self.time, _ = self._get_tick_status()

        # first we check for pnl coming in from a passive order being filled
        self._delayed_fill()
        reward = self.delayed_reward
        self.delayed_reward = 0

        # now execute the trade ... will will return 0 reward if the order is not accepted by the matchine engine
        reward += self._execute_trade(action)

        self.position, self.cost = self._get_position()
        self.vwap_m = self._calc_vwap()
        self.pending, self.pending_orders = self._get_pending()

        # get the state
        obs = self._get_state()

        # check if we are done
        if self.time >= self.end_time:
            done, info = True, 'trading is done.\t'
            info += 'timeout: {}\tinventory met: {}'.format(self.time == self.end_time,
                                                            self.position == self.inventory)
            reward += self._calc_reward(self.inventory, self.cost, portion=1, done=True)
        else:
            done, info = False, 'trading continues.'

        return obs, reward, done, info

    def _delayed_helper(self, order):
        oid = order['order_id']

        # get the order details
        filled, vwap, status = self._get_order(oid)

        if filled != order['filled'] and filled is not None:
            executed_quantity = abs(filled - order['filled'])

            # score the delayed reward
            self.delayed_reward += self._calc_reward(quantity=executed_quantity,
                                                     vwap_e=vwap,
                                                     portion=executed_quantity / order['quantity'],
                                                     done=False)

            # update the record
            self.order_history.loc[self.order_history['order_id'] == oid, 'i_filled'] = self.time
            self.order_history.loc[self.order_history['order_id'] == oid, 'filled'] = filled
            self.order_history.loc[self.order_history['order_id'] == oid, 'vwap'] = vwap
            self.order_history.loc[self.order_history['order_id'] == oid, 'active'] = status == 'OPEN'

    def _delayed_fill(self):
        # deal with the active orders
        active_orders = self.order_history.loc[self.order_history['active']]
        active_orders.apply(lambda order:
                            self._delayed_helper(order), axis=1)

        # update our position
        self.position, self.cost = self._get_position()

    def _execute_trade(self, action):
        trade_type, price, quantity = action[0], action[1], action[2]
        reward, status = 0, 'cancelled'

        if quantity == 0:       # if there is no quantity then don't bother to even submit the order
            trade_type = 2

        if trade_type == 0:     # limit order
            status, order_id, quantity_filled, vwap_e, tick = self._post_order('LIMIT',
                                                                               quantity,
                                                                               price,
                                                                               'SELL' if self.direction == 1 else 'BUY')
        elif trade_type == 1:   # market order
            self._cancel_orders()
            
            # update the quantity again to ensure that we execute the correct amount
            self.position, self.cost = self._get_position()
            quantity = abs(self.inventory - self.position)

            status, order_id, quantity_filled, vwap_e, tick = self._post_order('MARKET',
                                                                               quantity,
                                                                               0,
                                                                               'SELL' if self.direction == 1 else 'BUY')
        elif trade_type == 2:   # hold
            return 0
        else:                   # cancel
            self._cancel_orders()
            return 0

        if status not in ['failed', 'cancelled']:
            # add the order to the tracking list
            order_details = dict(zip(OH_COLS, [order_id,
                                               tick,
                                               tick if quantity_filled > 0 else -1,
                                               -1,
                                               price,
                                               quantity,
                                               quantity_filled,
                                               vwap_e,
                                               status == 'OPEN']))
            self.order_history = self.order_history.append(order_details, ignore_index=True)

            # calculate the reward
            reward = self._calc_reward(price*quantity, vwap_e, portion=1, done=False)

        return reward

    def _calc_reward(self, quantity, vwap_e, portion=1, done=False):
        vwap_m, reward = self._calc_vwap(), 0

        if vwap_e is not None:
            if self.reward_signal['collection'] == 'terminal':
                reward = quantity * (vwap_e - vwap_m) * self.direction if done else 0
            else:
                if self.reward_signal['method'] == 'VWAP_PNL':
                    reward = quantity * vwap_m * -self.direction if done else quantity * vwap_e * self.direction
                elif self.reward_signal['method'] == 'VWAP_TARGET':
                    reward = 0 if done else (vwap_e - vwap_m) * self.direction * portion

        return reward

    ## *****************************************************************************************************************
    # Functions that interact with RIT
    ## *****************************************************************************************************************

    def _get_trader_id(self):
        resp = s.get(self.api.format('trader'))

        if resp.ok:
            case = resp.json()
            return case['trader_id']
        else:
            return 'Matthew'

    def _get_tick_status(self):
        resp = s.get(self.api.format('case'))

        if resp.ok:
            case = resp.json()
            return case['status'], case['tick'], case['ticks_per_period']
        else:
            return 'PAUSED', 0, 300

    def _get_position(self):
        params = {'ticker': self.ticker}
        resp = s.get(self.api.format('securities'), params=params)

        if resp.ok:
            securities = resp.json()
            current_position, cost = abs(securities[0]['position']), securities[0]['vwap']

            return current_position, cost
        else:
            return None, None

    def _get_pending(self):
        params = {'status': 'OPEN'}
        resp = s.get(self.api.format('orders'), params=params)

        if resp.ok:
            pending_orders = resp.json()

            if len(pending_orders) == 0:
                outstanding_volume, outstanding_orders = 0, None
            else:
                outstanding_volume = sum([order['quantity'] - order['quantity_filled']
                                          for order in pending_orders if order['trader_id'] == self.trader_id])
                outstanding_orders = pd.DataFrame(pending_orders)
        else:
            outstanding_volume, outstanding_orders = 0, None

        return outstanding_volume, outstanding_orders

    def _get_order(self, order_id):
        resp = s.get(self.api.format('orders/{}'.format(str(order_id))))

        if resp.ok:
            order = resp.json()
            return order['quantity_filled'], order['vwap'], order['status']
        else:
            return None, None, None

    def _post_order(self, order_type, quantity, price, direction):
        params = {'ticker': self.ticker,
                  'type': order_type,
                  'quantity': quantity,
                  'price': price,
                  'action': direction}

        # post the order and return results
        resp = s.post(self.api.format('orders'), params=params)

        if resp.ok:
            details = resp.json()
            status, order_id, tick = details['status'], details['order_id'], details['tick']
            quantity_filled, vwap_e = details['quantity_filled'], details['vwap']
        else:
            status, order_id, tick = 'failed', -1, -1
            quantity_filled, vwap_e = -1, -1

        return status, order_id, quantity_filled, vwap_e, tick

    def _cancel_orders(self):
        # post the cancel request
        s.post(self.api.format('commands/cancel'), params={'ticker': self.ticker})

        time.sleep(0.05)

        # update the pending trackers and the order history
        self.pending, self.pending_orders = self._get_pending()
        self.order_history.loc[self.order_history['active'] == True, 'i_cancelled'] = self.time
        self.order_history['active'] = False

    def _calc_vwap(self):
        params = {'ticker': self.ticker}
        resp = s.get(self.api.format('securities/tas'), params=params)

        if resp.ok:
            quantity, weighted_avg_num = 0, 0

            for tas in resp.json():
                if tas['tick'] >= self.start_time:
                    quantity += tas['quantity']
                    weighted_avg_num += tas['price'] * tas['quantity']

            if len(resp.json()) == 0:
                return None
            else:
                try:
                    return weighted_avg_num / quantity
                except:
                    return 0

    def _get_lob(self):
        params = {'ticker': self.ticker}
        resp = s.get(self.api.format('securities/book'), params=params)

        # initialize the dataframe
        order_book = None

        if resp.ok:
            book = resp.json()

            try:
                for side in book:
                    quotes = pd.DataFrame(book[side])

                    # only care for the prices and the quantities
                    quotes.drop(columns=[col for col in quotes.columns
                                         if col not in ['price', 'quantity', 'quantity_filled']], inplace=True)

                    # find outstanding volumes
                    outstanding = quotes.groupby('price').agg({'quantity': 'sum',
                                                               'quantity_filled': 'sum'})
                    outstanding['volume'] = outstanding.apply(lambda quote:
                                                              quote['quantity'] - quote['quantity_filled'], axis=1)
                    outstanding.drop(columns=['quantity', 'quantity_filled'], inplace=True)

                    # flag whether the quote is a bid or an offer
                    outstanding['side'] = side

                    if order_book is None:
                        order_book = outstanding
                    else:
                        order_book = order_book.append(outstanding)
            except:
                order_book = pd.DataFrame(columns=['volume', 'side']).set_index('price')

        return order_book
