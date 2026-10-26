-- Generated automatically from UML model uml7.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "user" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "ticket" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "title" CHARACTER VARYING(255) NOT NULL
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('Open', 'InProgress', 'Resolved', 'Closed'))
  , "priority" CHARACTER VARYING(255) NOT NULL CHECK ("priority" IN ('Low', 'Medium', 'High', 'Critical'))
  , "creation_date" TIMESTAMP NOT NULL
  , "description" CHARACTER VARYING(255) NOT NULL
  , "user_id" CHARACTER VARYING(255) NOT NULL
  , "technician_id" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "comment" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "text" CHARACTER VARYING(255) NOT NULL
  , "date" TIMESTAMP NOT NULL
  , "ticket_id" BIGINT NOT NULL
);

CREATE TABLE "technician" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "email" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "ticket"
    ADD CONSTRAINT "fk_ticket_user"
    FOREIGN KEY ("user_id")
    REFERENCES "user" ("id");


ALTER TABLE "ticket"
    ADD CONSTRAINT "fk_ticket_technician"
    FOREIGN KEY ("technician_id")
    REFERENCES "technician" ("id");


ALTER TABLE "comment"
    ADD CONSTRAINT "fk_comment_ticket"
    FOREIGN KEY ("ticket_id")
    REFERENCES "ticket" ("id");


