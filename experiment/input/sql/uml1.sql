-- Generated automatically from UML model uml1.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "student" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "email" CHARACTER VARYING(255) NOT NULL
  , "age" INTEGER NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "course" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "title" CHARACTER VARYING(255) NOT NULL
  , "credits" INTEGER NOT NULL
);

CREATE TABLE "registration" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "date" TIMESTAMP NOT NULL
  , "student_id" BIGINT NOT NULL
  , "course_id" BIGINT NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "registration"
    ADD CONSTRAINT "fk_registration_student"
    FOREIGN KEY ("student_id")
    REFERENCES "student" ("id");


ALTER TABLE "registration"
    ADD CONSTRAINT "fk_registration_course"
    FOREIGN KEY ("course_id")
    REFERENCES "course" ("id");


