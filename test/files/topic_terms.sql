--
-- PostgreSQL database dump
--

-- Dumped from database version 18.1 (Debian 18.1-1.pgdg13+2)
-- Dumped by pg_dump version 18.1 (Debian 18.1-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: table_topic_term_b6f63c93-7ed0-4320-9957-dbd3ec690624; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public."topic_terms" (
    topic_id bigint,
    term text,
    weight double precision
);


ALTER TABLE public."topic_terms" OWNER TO postgres;

--
-- Data for Name: table_topic_term_b6f63c93-7ed0-4320-9957-dbd3ec690624; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public."topic_terms" VALUES (0, 'bike', 59.1918460791726);
INSERT INTO public."topic_terms" VALUES (0, 'accident', 53.20056393232391);
INSERT INTO public."topic_terms" VALUES (0, 'road', 50.48598957803687);
INSERT INTO public."topic_terms" VALUES (0, 'car', 45.94439645952022);
INSERT INTO public."topic_terms" VALUES (0, 'patient', 44.20207675328453);
INSERT INTO public."topic_terms" VALUES (0, 'safety', 42.201067986182665);
INSERT INTO public."topic_terms" VALUES (0, 'injury', 39.199803727883875);
INSERT INTO public."topic_terms" VALUES (0, 'study', 36.58007376096543);
INSERT INTO public."topic_terms" VALUES (0, 'cyclist', 32.505037925761734);
INSERT INTO public."topic_terms" VALUES (0, 'factor', 29.200734580467568);
INSERT INTO public."topic_terms" VALUES (1, 'car', 87.45624160202024);
INSERT INTO public."topic_terms" VALUES (1, 'bike', 83.20743256160291);
INSERT INTO public."topic_terms" VALUES (1, 'crash', 75.20185154992718);
INSERT INTO public."topic_terms" VALUES (1, 'bicycle', 73.02224809650083);
INSERT INTO public."topic_terms" VALUES (1, 'vehicle', 70.41451826492317);
INSERT INTO public."topic_terms" VALUES (1, 'road', 65.91519655863446);
INSERT INTO public."topic_terms" VALUES (1, 'accident', 53.199110287474916);
INSERT INTO public."topic_terms" VALUES (1, 'traffic', 44.200033079638864);
INSERT INTO public."topic_terms" VALUES (1, 'datum', 43.20065093486357);
INSERT INTO public."topic_terms" VALUES (1, 'cyclist', 40.895451225540576);
INSERT INTO public."topic_terms" VALUES (2, 'accident', 52.20255139113972);
INSERT INTO public."topic_terms" VALUES (2, 'traffic', 31.201392651780292);
INSERT INTO public."topic_terms" VALUES (2, 'safety', 30.201099323866146);
INSERT INTO public."topic_terms" VALUES (2, 'car', 30.20023290922999);
INSERT INTO public."topic_terms" VALUES (2, 'study', 27.20079045712977);
INSERT INTO public."topic_terms" VALUES (2, 'bicycle', 27.20068525242122);
INSERT INTO public."topic_terms" VALUES (2, 'behavior', 25.201600668144895);
INSERT INTO public."topic_terms" VALUES (2, 'cyclist', 22.200778603045276);
INSERT INTO public."topic_terms" VALUES (2, 'model', 20.20093139254029);
INSERT INTO public."topic_terms" VALUES (2, 'factor', 20.200886378333102);
INSERT INTO public."topic_terms" VALUES (3, 'injury', 82.20278429960825);
INSERT INTO public."topic_terms" VALUES (3, 'bike', 42.19901662997125);
INSERT INTO public."topic_terms" VALUES (3, 'child', 39.20171975517025);
INSERT INTO public."topic_terms" VALUES (3, 'accident', 39.20007478047872);
INSERT INTO public."topic_terms" VALUES (3, 'car', 39.1996141202332);
INSERT INTO public."topic_terms" VALUES (3, 'risk', 36.20035598056616);
INSERT INTO public."topic_terms" VALUES (3, 'crash', 33.2005715769374);
INSERT INTO public."topic_terms" VALUES (3, 'school', 33.20041352502453);
INSERT INTO public."topic_terms" VALUES (3, 'year', 32.20110140561873);
INSERT INTO public."topic_terms" VALUES (3, 'study', 31.199843028446885);
INSERT INTO public."topic_terms" VALUES (4, 'bike', 90.20366648216675);
INSERT INTO public."topic_terms" VALUES (4, 'bicycle', 39.200579142722354);
INSERT INTO public."topic_terms" VALUES (4, 'vehicle', 37.20105190777457);
INSERT INTO public."topic_terms" VALUES (4, 'speed', 36.20171810689791);
INSERT INTO public."topic_terms" VALUES (4, 'car', 34.199514908977534);
INSERT INTO public."topic_terms" VALUES (4, 'study', 31.200051732299197);
INSERT INTO public."topic_terms" VALUES (4, 'road', 29.199793490140163);
INSERT INTO public."topic_terms" VALUES (4, 'risk', 27.2001862345917);
INSERT INTO public."topic_terms" VALUES (4, 'use', 25.200710833296004);
INSERT INTO public."topic_terms" VALUES (4, 'driver', 25.20051046618428);


--
-- PostgreSQL database dump complete
--

