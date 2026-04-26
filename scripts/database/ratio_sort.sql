DROP TABLE sigma_ci_1_sigma_iv_1_Eg_25;
CREATE TABLE sigma_ci_1_sigma_iv_1_Eg_25 AS
SELECT * FROM TOTAL
WHERE sigma_opt_ci = '1e-13'
AND sigma_opt_iv = '1e-13'
AND CB_E = 2.5
AND folder_path LIKE '/home/duncan/data/main/archive/V_6.5/Eg_2.5/mu_I%';

DROP TABLE sigma_ci_1_sigma_iv_5_Eg_25;
CREATE TABLE sigma_ci_1_sigma_iv_5_Eg_25 AS
SELECT * FROM TOTAL
WHERE sigma_opt_ci = '1e-13'
AND sigma_opt_iv = '5e-13'
AND CB_E = 2.5
AND folder_path LIKE '/home/duncan/data/main/archive/V_6.5/Eg_2.5/mu_I%';

DROP TABLE sigma_ci_5_sigma_iv_1_Eg_25;
CREATE TABLE sigma_ci_5_sigma_iv_1_Eg_25 AS
SELECT * FROM TOTAL
WHERE sigma_opt_ci = '5e-13'
AND sigma_opt_iv = '1e-13'
AND CB_E = 2.5
AND folder_path LIKE '/home/duncan/data/main/archive/V_6.5/Eg_2.5/mu_I%';

DROP TABLE sigma_ci_5_sigma_iv_5_Eg_25;
CREATE TABLE sigma_ci_5_sigma_iv_5_Eg_25 AS
SELECT * FROM TOTAL
WHERE sigma_opt_ci = '5e-13'
AND sigma_opt_iv = '5e-13'
AND CB_E = 2.5
AND folder_path LIKE '/home/duncan/data/main/archive/V_6.5/Eg_2.5/mu_I%';

