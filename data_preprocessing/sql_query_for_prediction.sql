(SELECT 
    "MSISDN",
    AVG("TOTAL_REVENUE" ) AS "WEIGHTED_AVG_TOTAL_REVENUE",
    AVG("TOTAL_MIN" ) AS "WEIGHTED_AVG_TOTAL_MIN",
    AVG("TOTAL_SMS_Qty" ) AS "WEIGHTED_AVG_TOTAL_SMS_Qty", 
    AVG("TOTAL_VOLUME_MB" ) AS "WEIGHTED_AVG_TOTAL_VOLUME_MB", 
    AVG("TOTAL_BUNDLE_VOLUME_MB" ) AS "WEIGHTED_AVG_TOTAL_BUNDLE_VOLUME_MB", 
    AVG("CALL_ONNET_REV" ) AS "WEIGHTED_AVG_CALL_ONNET_REV", 
    AVG("CALL_ONNET_DURATION_MIN"  ) AS "WEIGHTED_AVG_CALL_ONNET_DURATION_MIN", 
    AVG("Call_OnNet_QTY" ) AS "WEIGHTED_AVG_Call_OnNet_QTY", 
    AVG("CALL_OFFNET_DURATION_MIN" ) AS "WEIGHTED_AVG_CALL_OFFNET_DURATION_MIN", 
    AVG("Call_OffNet_QTY" ) AS "WEIGHTED_AVG_Call_OffNet_QTY", 
    AVG("CALL_OFFNET_BEELINE_FIX_DURATION_MIN" ) AS "WEIGHTED_AVG_CALL_OFFNET_BEELINE_FIX_DURATION_MIN", 
    AVG("CALL_OFFNET_BEELINE_GSM_DURATION_MIN" ) AS "WEIGHTED_AVG_CALL_OFFNET_BEELINE_GSM_DURATION_MIN", 
    AVG("CALL_OFFNET_ORANGE_DURATION_MIN" ) AS "WEIGHTED_AVG_CALL_OFFNET_ORANGE_DURATION_MIN", 
    AVG("PAY_QTY" ) AS "WEIGHTED_AVG_PAY_QTY", 
    AVG("PAY_AMOUNT" ) AS "WEIGHTED_AVG_PAY_AMOUNT", 
    AVG("NEGBAL_QTY" ) AS "WEIGHTED_AVG_NEGBAL_QTY", 
    AVG("Call_MT_OnNet_min" ) AS "WEIGHTED_AVG_Call_MT_OnNet_min", 
    AVG("Call_MT_OnNet_GSM_min") AS "WEIGHTED_AVG_Call_MT_OnNet_GSM_min",
    AVG("Call_MT_OnNet_QTY" ) AS "WEIGHTED_AVG_Call_MT_OnNet_QTY", 
    AVG("Call_MT_OffNet_Mob_min") AS "WEIGHTED_AVG_Call_MT_OffNet_Mob_min", 
    AVG("Call_MT_OffNet_Mob_BeeGSM_min" ) AS "WEIGHTED_AVG_Call_MT_OffNet_Mob_BeeGSM_min", 
    AVG("Call_MT_OffNet_Mob_OraGSM_min" ) AS "WEIGHTED_AVG_Call_MT_OffNet_Mob_OraGSM_min", 
    AVG("Call_MT_OffNet_Mob_QTY" ) AS "WEIGHTED_AVG_Call_MT_OffNet_Mob_QTY" 
FROM 
    (SELECT * 
           
    FROM 
        (SELECT "MSISDN",
                "TOTAL_REVENUE", "TOTAL_MIN", "TOTAL_SMS_Qty", "TOTAL_VOLUME_MB",
                "TOTAL_BUNDLE_VOLUME_MB", "CALL_ONNET_REV", "CALL_ONNET_DURATION_MIN", 
                "Call_OnNet_QTY", "CALL_OFFNET_DURATION_MIN", "Call_OffNet_QTY", 
                "CALL_OFFNET_BEELINE_FIX_DURATION_MIN", "CALL_OFFNET_BEELINE_GSM_DURATION_MIN", 
                "CALL_OFFNET_ORANGE_DURATION_MIN", "PAY_QTY", "PAY_AMOUNT", "NEGBAL_QTY", 
                "Call_MT_OnNet_min", "Call_MT_OnNet_GSM_min", "Call_MT_OnNet_QTY", 
                "Call_MT_OffNet_Mob_min", "Call_MT_OffNet_Mob_BeeGSM_min", 
                "Call_MT_OffNet_Mob_OraGSM_min", "Call_MT_OffNet_Mob_QTY"
           FROM "test_data"
           WHERE "DATE_ID" BETWEEN :start_date AND :end_date


                AND ("TOTAL_REVENUE" IS NOT NULL) 
                AND ("TOTAL_MIN" IS NOT NULL) 
                AND ("TOTAL_SMS_Qty" IS NOT NULL) 
                AND ("TOTAL_VOLUME_MB" IS NOT NULL) 
                AND ("TOTAL_BUNDLE_VOLUME_MB" IS NOT NULL) 
                AND ("CALL_ONNET_REV" IS NOT NULL) 
                AND ("CALL_ONNET_DURATION_MIN" IS NOT NULL) 
                AND ("Call_OnNet_QTY" IS NOT NULL) 
                AND ("CALL_OFFNET_DURATION_MIN" IS NOT NULL) 
                AND ("Call_OffNet_QTY" IS NOT NULL) 
                AND ("CALL_OFFNET_BEELINE_FIX_DURATION_MIN" IS NOT NULL) 
                AND ("CALL_OFFNET_BEELINE_GSM_DURATION_MIN" IS NOT NULL) 
                AND ("CALL_OFFNET_ORANGE_DURATION_MIN" IS NOT NULL) 
                AND ("PAY_QTY" IS NOT NULL) 
                AND ("PAY_AMOUNT" IS NOT NULL) 
                AND ("NEGBAL_QTY" IS NOT NULL) 
                AND ("Call_MT_OnNet_min" IS NOT NULL) 
                AND ("Call_MT_OnNet_GSM_min" IS NOT NULL) 
                AND ("Call_MT_OnNet_QTY" IS NOT NULL) 
                AND ("Call_MT_OffNet_Mob_min" IS NOT NULL) 
                AND ("Call_MT_OffNet_Mob_BeeGSM_min" IS NOT NULL) 
                AND ("Call_MT_OffNet_Mob_OraGSM_min" IS NOT NULL) 
                AND ("Call_MT_OffNet_Mob_QTY" IS NOT NULL) 
                AND "SUB_TYPE" = 'PREPAID' 
                AND ("Only_CallMT" + "Only_Payment" + "Only_CreditClearance" + "Only_Debt" = 0)
                AND "TP" in  ('Viva 1500', 'Viva 2500', 'Viva 3500', 'Viva 5500',
                            'New Youth X', 'New Youth Y', 'New Youth Z', 'Student-Yes', 
                            'Menq', 'Dialect') ) as "c") as "d"
                  
        GROUP BY "MSISDN")