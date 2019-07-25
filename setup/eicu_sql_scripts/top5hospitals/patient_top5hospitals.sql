-- take all patients from the 5 largest hospitals

DROP MATERIALIZED VIEW IF EXISTS patient_top5hospitals CASCADE;
CREATE MATERIALIZED VIEW patient_top5hospitals as (
  with top_hospitals as (
    select hospitalid, count(patientunitstayid) as n
    from patient
    group by hospitalid
    order by n desc
    limit 5
    )
    select *
    from patient
           join (
      select hospitalid as top5hospitalid from top_hospitals
    ) as tt
          on tt.top5hospitalid = patient.hospitalid
);