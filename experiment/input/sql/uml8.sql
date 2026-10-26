-- Generated automatically from UML model uml8.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "employee" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "salary" REAL NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
  , "hire_date" TIMESTAMP NOT NULL
  , "department_id" CHARACTER VARYING(255) NOT NULL
  , "position_id" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "department" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "location" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "position" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "title" CHARACTER VARYING(255) NOT NULL
  , "description" CHARACTER VARYING(255) NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "employee"
    ADD CONSTRAINT "fk_employee_department"
    FOREIGN KEY ("department_id")
    REFERENCES "department" ("id");


ALTER TABLE "employee"
    ADD CONSTRAINT "fk_employee_position"
    FOREIGN KEY ("position_id")
    REFERENCES "position" ("id");


