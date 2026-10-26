-- Generated automatically from UML model uml6.
-- SQL/Foundation-oriented DDL using standard delimited identifiers.
-- UML-to-relational mapping choices are explicit transformation policies;
-- ISO/IEC 9075 defines SQL, not a normative UML-to-SQL mapping.

CREATE TABLE "guest" (
    "id" CHARACTER VARYING(255) PRIMARY KEY
  , "name" CHARACTER VARYING(255) NOT NULL
  , "email" CHARACTER VARYING(255) NOT NULL
  , "phone" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "hotel" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "address" CHARACTER VARYING(255) NOT NULL
  , "name" CHARACTER VARYING(255) NOT NULL
);

CREATE TABLE "reservation" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "total_price" REAL NOT NULL
  , "status" CHARACTER VARYING(255) NOT NULL CHECK ("status" IN ('Pending', 'Confirmed', 'Cancelled', 'Completed'))
  , "check_in" TIMESTAMP NOT NULL
  , "check_out" TIMESTAMP NOT NULL
  , "guest_id" CHARACTER VARYING(255) NOT NULL
  , "room_id" BIGINT NOT NULL
);

CREATE TABLE "room" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
  , "price_per_night" REAL NOT NULL
  , "type" CHARACTER VARYING(255) NOT NULL CHECK ("type" IN ('Single', 'Double', 'Suite'))
  , "capacity" INTEGER NOT NULL
  , "number" INTEGER NOT NULL
  , "hotel_id" BIGINT NOT NULL
);


-- Tables for multivalued primitive attributes

-- Junction tables for many-to-many associations

-- Foreign-key constraints for one-to-many and one-to-one associations
ALTER TABLE "reservation"
    ADD CONSTRAINT "fk_reservation_guest"
    FOREIGN KEY ("guest_id")
    REFERENCES "guest" ("id");


ALTER TABLE "reservation"
    ADD CONSTRAINT "fk_reservation_room"
    FOREIGN KEY ("room_id")
    REFERENCES "room" ("id");


ALTER TABLE "room"
    ADD CONSTRAINT "fk_room_hotel"
    FOREIGN KEY ("hotel_id")
    REFERENCES "hotel" ("id");


