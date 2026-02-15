import React from 'react'
import ReactDOM from 'react-dom/client'
import 'flexlayout-react/style/light.css'
import AppShell from './app/AppShell'
import './styles/app.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppShell />
  </React.StrictMode>
)
