#metaoptimization

s.p_min = 0.0233146;
s.p_max = 1.79131;
s.M = 10;
s.N = 11;
s.n_min = 2;
s.n_max = 2;
s.m_min = 2;
s.m_max = 3;
s.k = 8;
s.l = 2;
s.ep_elite = 0.07721;
s.ep_dead = 0.282488;
s.max_mutation = 10;
s.gray_percent = 0.662659;
s.p_motion = 0.226807;
s.p_trade = 0.451844;
s.p_war = 0.150414;
s.p_epidemic = 0.0780067;
s.p_migration = 0.0929283;
s.adaptive_actions = true;
s.action_alpha = 0.321974;
s.action_pmin = 0.05;
s.action_warmup_frac = 0.00759028;
s.stagnation_limit = 17;
s.restart_country_frac = 0.236802;
s.migration_frac = 0.212646;



| ID  | Название функции                             | Класс           | Успех (%) | Avg Error | Std Dev  | Best Error | Среднее число вызовов |
| --- | -------------------------------------------- | --------------- | --------- | --------- | -------- | ---------- | --------------------- |
| F1  | Shifted and Rotated Bent Cigar               | Унимодальная    | 0%        | 2383.60   | 2590.15  | 5.67172    | 100064.0              |
| F2  | Shifted and Rotated Sum of Different Power   | Унимодальная    | 100%      | 0.00      | 0.00     | 0.00       | 100043.0              |
| F3  | Shifted and Rotated Zakharov                 | Унимодальная    | 100%      | 1.14e-13  | 8.14e-14 | 0.00       | 100051.0              |
| F4  | Shifted and Rotated Rosenbrock               | Мультимодальная | 0%        | 2.42496   | 1.53049  | 1.70e-03   | 100051.0              |
| F5  | Shifted and Rotated Rastrigin                | Мультимодальная | 0%        | 19.0696   | 7.24495  | 6.96471    | 100050.0              |
| F6  | Shifted and Rotated Expanded Scaffer F6      | Мультимодальная | 80%       | 9.54e-04  | 3.73e-03 | 1.48e-12   | 100050.0              |
| F7  | Shifted and Rotated Lunacek Bi_Rastrigin     | Мультимодальная | 0%        | 32.3615   | 9.49813  | 17.4811    | 99970.6               |
| F8  | Shifted and Rotated Non-Continuous Rastrigin | Мультимодальная | 0%        | 18.6057   | 6.38712  | 8.95463    | 100057.0              |
| F9  | Shifted and Rotated Levy                     | Мультимодальная | 35%       | 4.51102   | 17.5702  | 0.00       | 100060.0              |
| F10 | Shifted and Rotated Schwefel                 | Мультимодальная | 0%        | 511.942   | 229.700  | 121.916    | 99354.7               |
| F11 | Hybrid Function 1 ($N=3$)                    | Гибридная       | 0%        | 16.8752   | 9.84277  | 3.17254    | 99667.6               |
| F12 | Hybrid Function 2 ($N=3$)                    | Гибридная       | 0%        | 11480.3   | 15513.9  | 1031.34    | 100048.0              |
| F13 | Hybrid Function 3 ($N=3$)                    | Гибридная       | 0%        | 75.6123   | 102.233  | 7.46385    | 100045.0              |
| F14 | Hybrid Function 4 ($N=4$)                    | Гибридная       | 0%        | 29.9217   | 11.6528  | 6.46351    | 100059.0              |
| F15 | Hybrid Function 5 ($N=4$)                    | Гибридная       | 0%        | 12.4144   | 11.0632  | 2.40069    | 100040.0              |
| F16 | Hybrid Function 6 ($N=4$)                    | Гибридная       | 0%        | 100.945   | 112.709  | 0.211388   | 100062.0              |
| F17 | Hybrid Function 7 ($N=5$)                    | Гибридная       | 0%        | 48.4627   | 30.9294  | 6.61383    | 100049.0              |
| F18 | Hybrid Function 8 ($N=5$)                    | Гибридная       | 0%        | 680.533   | 1326.28  | 15.3198    | 100050.0              |
| F19 | Hybrid Function 9 ($N=5$)                    | Гибридная       | 0%        | 5.00398   | 3.49560  | 1.21645    | 100046.0              |
| F20 | Hybrid Function 10 ($N=6$)                   | Гибридная       | 0%        | 24.9951   | 17.6536  | 0.624348   | 99957.9               |
| F21 | Composition Function 1 ($N=3$)               | Композитная     | 0%        | 119.900   | 44.4760  | 100.000    | 99626.5               |
| F22 | Composition Function 2 ($N=3$)               | Композитная     | 0%        | 82.9679   | 33.1441  | 19.8069    | 99978.2               |
| F23 | Composition Function 3 ($N=4$)               | Композитная     | 0%        | 325.136   | 8.85404  | 313.169    | 100056.0              |
| F24 | Composition Function 4 ($N=4$)               | Композитная     | 0%        | 188.306   | 120.374  | 100.000    | 100053.0              |
| F25 | Composition Function 5 ($N=5$)               | Композитная     | 0%        | 412.706   | 75.2608  | 100.052    | 100051.0              |
| F26 | Composition Function 6 ($N=5$)               | Композитная     | 5%        | 323.776   | 102.074  | 3.92e-10   | 100065.0              |
| F27 | Composition Function 7 ($N=6$)               | Композитная     | 0%        | 393.127   | 2.76256  | 389.638    | 100051.0              |
| F28 | Composition Function 8 ($N=6$)               | Композитная     | 0%        | 406.081   | 114.002  | 300.000    | 100062.0              |
| F29 | Composition Function 9 ($N=3$)               | Композитная     | 0%        | 293.879   | 35.0086  | 233.579    | 100052.0              |
| F30 | Composition Function 10 ($N=3$)              | Композитная     | 0%        | 3324.19   | 4630.10  | 550.520    | 99863.2               |