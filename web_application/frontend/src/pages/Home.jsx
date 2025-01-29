import React from 'react';
import { Link, NavLink } from 'react-router-dom';
import "../styles/home.css";

export default function Home() {

	return (

		<div className="home-container">
			<div className="logo">

			</div>
			<div className="home-title">

			</div>
			<div className="plate-type-selection">
				<NavLink to="/well_selection">
					
				</NavLink>
			</div>
		</div>
	); 
}