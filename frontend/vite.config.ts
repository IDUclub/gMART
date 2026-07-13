import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(({command})=>({base:command==="build"?"/ui/":"/",plugins:[react()],server:{proxy:{"/api-agents":{target:"http://127.0.0.1:80",changeOrigin:true,rewrite:p=>p.replace(/^\/api-agents/,"")},"/api-chats":{target:"http://127.0.0.1:8010",changeOrigin:true,rewrite:p=>p.replace(/^\/api-chats/,"")}}}}));
