-- Generated automatically from UML model uml4.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "book" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "title" CHARACTER VARYING(255) NOT NULL
  , "year" INTEGER NOT NULL
  , "author" CHARACTER VARYING(255) NOT NULL
  , "isbn" CHARACTER VARYING(255) NOT NULL
  , "publisher_id" BIGINT NOT NULL
);

CREATE TABLE "member" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "active" BOOLEAN NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "publisher" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "country" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "loan" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('Active', 'Returned', 'Overdue'))
  , "start_date" TIMESTAMP NOT NULL
  , "due_date" TIMESTAMP NOT NULL
  , "return_date" TIMESTAMP NOT NULL
  , "member_id" CHARACTER VARYING(255) NOT NULL
  , "book_id" BIGINT NOT NULL
);


-- Tables for multivalued primitive attributes
CREATE TABLE "book_categories" (
    "book_id" BIGINT NOT NULL
  , "value" CHARACTER VARYING(255) NOT NULL
  , PRIMARY KEY ("book_id", "value")
  , CONSTRAINT "fk_book_categories"
        FOREIGN KEY ("book_id")
        REFERENCES "book" ("id")
);



-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "book"
    ADD CONSTRAINT "fk_book_publisher"
    FOREIGN KEY ("publisher_id")
    REFERENCES "publisher" ("id");


ALTER TABLE "loan"
    ADD CONSTRAINT "fk_loan_member"
    FOREIGN KEY ("member_id")
    REFERENCES "member" ("id");


ALTER TABLE "loan"
    ADD CONSTRAINT "fk_loan_book"
    FOREIGN KEY ("book_id")
    REFERENCES "book" ("id");


