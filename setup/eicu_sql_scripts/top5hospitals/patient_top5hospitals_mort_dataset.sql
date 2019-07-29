DROP MATERIALIZED VIEW IF EXISTS patient_top5hospitals_mort_dataset CASCADE;
CREATE MATERIALIZED VIEW patient_top5hospitals_mort_dataset as (
    select  patientunitstayid, hosp_mort from icustay_detail
    where hospitalid in (
        select distinct hospitalid  from patient_top5hospitals
    )
    and
    icu_los_hours >= 26
);