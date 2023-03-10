from datetime import date, timedelta

import holidays
import numpy as np
from gym import Env, spaces

class DigitalAdvertisingEnv(Env):
    """Class for representing the digital advertising environment

    Observation:
        The observation will be in the form of a Tuple e.g. (0,3,50), where
        the first value is 0 or 1, denoting if the date is a holiday, the second value
        is the day of the week, 0 is Monday to 6 being Sunday, and the third value is
        the week of the year, from 0 to 51.

    Note: the final CTR value that is returned is calculated with the following method:
    1. Based on the "today" date, determine which modifiers are applicable
    2. Adjust the `base` distribution parameters with the relevant modifiers by averaging the parameters
    3. Sample from the adjusted Beta distribution
    4. Return final CTR value with adjustment

    For example:
    1. Given that "today" is Monday Dec 25, the relevant modifiers are `mon`, `holiday` and `w50`, 
        with the respective values of {'mon' : (1,2), 'holiday' : (3,1), 'w50' : (5,1)}
    2. Adjust the `base` distribution with relevant modifiers, creatives, websites:
        A. `base` + `mon` -> (0.5,0.5) + (1,2) = ((0.5+1)/2, (0.5+2)/2) = (0.75,1.25)
        B. `A` + `holiday` -> (0.75,1.25) + (3,1) = ((0.75+3)/2,(1.25+1)/2) = (1.875,1.125)
        C. `B` + `w50` ->  (1.875,1.125) + (5,1) = ((1.875+5)/2, (1.125+1)/2) = (3.4375,1.0625)
    3. Sample from Beta(3.4375, 1.0625) -> 0.939
    4. return 0.939 / 10 = .0939 (which is a 9.39% CTR, note: the CTR ranges between 0% and 10%)
    """
    def __init__(self, creatives, websites, modifiers, base = (1,1), start_date = None):
        """Initializes the digital advertising environment

        :param creatives: distribution parameters for the creatives
        :type creatives: tuple/list with the values as a tuple of the Beta distribution parameters,
            e.g. [(1,1.5), (2,2)]
        :param websites: distribution parameters for the webisites
        :type websites: tuple/list with the values as a tuple of the Beta distribution parameters,
            e.g. [(1,1.5), (2,2)]
        :param modifiers: distribution parameters for the modifiers, currently the following modifiers
            are supported: day of week (`mon`, `tue`, etc.), week of year (`w0`, `w1`, etc.), holidays (`holiday`)
        :type modifiers: dict with the keys as the modifier and the value as a tuple
            e.g. {'mon' : (1,2), 'holiday' : (3,1), 'w50' : (5,1)}
        :param base: the base Beta distribution parameters to sample the CTR, defaults to (1,1)
        :type base: tuple of the form (<alpha>, <beta>), values are the distribution parameters
            e.g. (1,1)
        :param start_date: the start date for simulation, defaults to date(1990,1,1)
        :type start_date: datetime.date object, optional
        """         
        super(DigitalAdvertisingEnv, self).__init__()
        # Parameter assertions
        assert all([y > 0 for x in creatives for y in x]), f"Creative parameters not valid: {creatives}"
        assert all([y > 0 for x in websites for y in x]), f"Websites parameters not valid: {websites}"
        assert all([y > 0 for x in modifiers.values() for y in x]), f"Modifiers parameters not valid: {modifiers}"
        assert all([x > 0 for x in base]), f"Base parameters not valid: {base}"
        assert start_date is None or isinstance(start_date, date), f"Invalid date: {start_date}"
        # Setup variables
        self.creatives = creatives
        self.websites = websites
        self.modifiers = modifiers
        self.base = base
        self.start_date = date(1990,1,1) if start_date is None else start_date
        # Setup gym variables
        self.observation_space = spaces.MultiDiscrete([2, 7, 53])
        self.action_space = spaces.MultiDiscrete([len(creatives), len(websites)])

    def _is_holiday(self):
        return int(self.today in 
            holidays.UnitedStates(years = self.today.year).keys())
    
    def _get_day_of_week(self):
        return self.today.weekday()
    
    def _get_week_of_year(self):
        return self.today.isocalendar()[1] - 1

    def _get_obs(self):    
        self.today = self.today + timedelta(days = 1)
        output = (
            self._is_holiday(), 
            self._get_day_of_week(), 
            self._get_week_of_year(),
        )
        return output
    
    def _get_modifier_params(self):
        params = []
        # Day modifier
        days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        day_of_week = days[self._get_day_of_week()]
        if day_of_week in self.modifiers:
            params.append(self.modifiers[day_of_week])
        # Week modifier
        week_of_year = f"w{self._get_week_of_year()}"
        if week_of_year in self.modifiers:
            params.append(self.modifiers[week_of_year])
        # Holiday modifier
        if self._is_holiday() == 1 and 'holiday' in self.modifiers:
            params.append(self.modifiers['holiday'])
        return params

    def _get_action_params(self, action):
        return [self.creatives[action[0]], self.websites[action[1]]]

    def _calc_ctr(self, alpha, beta):
        return np.random.beta(alpha, beta) / 10

    def reset(self):
        self.today = self.start_date
        return self._get_obs()
    
    def step(self, action):
        # Get inputs
        modifiers = self._get_modifier_params()
        action_params = self._get_action_params(action)
        params = [*modifiers, *action_params]
        # Calculate Beta params
        alpha = np.mean([x[0] for x in params])
        beta = np.mean([x[1] for x in params])
        # Sample distribution
        reward = self._calc_ctr(alpha, beta)
        obs = self._get_obs()
        # Calculate total regret
        combos = [(x,y) for x in range(len(self.creatives)) 
            for y in range(len(self.websites)) 
            if x != action[0] and y != action[1]]
        best_reward = reward
        best_action = action
        for combo in combos:
            combo_params = self._get_action_params(combo)
            params = [*modifiers, *combo_params]
            alpha = np.mean([x[0] for x in params])
            beta = np.mean([x[1] for x in params])
            combo_reward = self._calc_ctr(alpha, beta)
            if combo_reward > best_reward:
                best_reward = combo_reward
                best_action = combo
        info = {
            'total_regret' : best_reward - reward, 
            'best_action' : best_action
            }
        return obs, reward, False, info