-- Generated automatically from UML model uml5.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "patient" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "phone" CHARACTER VARYING(255) NOT NULL
  , "birth_date" TIMESTAMP NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "doctor" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "specialty" CHARACTER VARYING(255) NOT NULL
  , "medical_center_id" BIGINT NOT NULL
);

CREATE TABLE "appointment" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "reason" CHARACTER VARYING(255) NOT NULL
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('Scheduled', 'Completed', 'Cancelled', 'NoShow'))
  , "date_time" TIMESTAMP NOT NULL
  , "patient_id" CHARACTER VARYING(255) NOT NULL
  , "doctor_id" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "medical_center" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "address" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "doctor"
    ADD CONSTRAINT "fk_doctor_medical_center"
    FOREIGN KEY ("medical_center_id")
    REFERENCES "medical_center" ("id");


ALTER TABLE "appointment"
    ADD CONSTRAINT "fk_appointment_patient"
    FOREIGN KEY ("patient_id")
    REFERENCES "patient" ("id");


ALTER TABLE "appointment"
    ADD CONSTRAINT "fk_appointment_doctor"
    FOREIGN KEY ("doctor_id")
    REFERENCES "doctor" ("id");


