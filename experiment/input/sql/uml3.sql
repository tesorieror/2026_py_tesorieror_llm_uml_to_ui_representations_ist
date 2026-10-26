-- Generated automatically from UML model uml3.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "customer" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "password" CHARACTER VARYING(255) NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
  , "mobile" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "invoice" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "date" TIMESTAMP NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
  , "vat" REAL NOT NULL
  , "address" CHARACTER VARYING(255) NOT NULL
  , "subtotal" REAL NOT NULL
  , "total" REAL NOT NULL
  , "number" INTEGER NOT NULL
  , "customer_id" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "product" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "description" CHARACTER VARYING(255) NOT NULL
  , "price" REAL NOT NULL
);

CREATE TABLE "line" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "unit_price" REAL NOT NULL
  , "quantity" REAL NOT NULL
  , "total" REAL NOT NULL
  , "description" CHARACTER VARYING(255) NOT NULL
  , "vat" REAL NOT NULL
  , "subtotal_price" REAL NOT NULL
  , "invoice_id" BIGINT NOT NULL
  , "product_id" BIGINT NOT NULL
);


-- Tables for multivalued primitive attributes
CREATE TABLE "customer_addresses" (
    "customer_id" CHARACTER VARYING(255) NOT NULL
  , "value" CHARACTER VARYING(255) NOT NULL
  , PRIMARY KEY ("customer_id", "value")
  , CONSTRAINT "fk_customer_addresses"
        FOREIGN KEY ("customer_id")
        REFERENCES "customer" ("id")
);



-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "invoice"
    ADD CONSTRAINT "fk_invoice_customer"
    FOREIGN KEY ("customer_id")
    REFERENCES "customer" ("id");


ALTER TABLE "line"
    ADD CONSTRAINT "fk_line_invoice"
    FOREIGN KEY ("invoice_id")
    REFERENCES "invoice" ("id");


ALTER TABLE "line"
    ADD CONSTRAINT "fk_line_product"
    FOREIGN KEY ("product_id")
    REFERENCES "product" ("id");


